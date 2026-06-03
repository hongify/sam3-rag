# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe
from typing import Tuple, Dict, Optional, List 

import numpy as np
import PIL
import PIL.Image as Image
import torch
from sam3.model import box_ops
from sam3.model.data_misc import FindStage, interpolate
from torchvision.transforms import v2
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2
from sam3.model.geometry_encoders import Prompt 


class Sam3Processor:
    """ """

    def __init__(self, model, resolution=1008, device="cuda", confidence_threshold=0.5):
        self.model = model
        self.resolution = resolution
        self.device = device
        self.transform = v2.Compose(
            [
                v2.ToDtype(torch.uint8, scale=True),
                v2.Resize(size=(resolution, resolution)),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        self.confidence_threshold = confidence_threshold

        self.find_stage = FindStage(
            img_ids=torch.tensor([0], device=device, dtype=torch.long),
            text_ids=torch.tensor([0], device=device, dtype=torch.long),
            input_boxes=None,
            input_boxes_mask=None,
            input_boxes_label=None,
            input_points=None,
            input_points_mask=None,
        )

    @torch.inference_mode()
    def set_image(self, image, state=None):
        """Sets the image on which we want to do predictions."""
        if state is None:
            state = {}

        if isinstance(image, PIL.Image.Image):
            width, height = image.size
        elif isinstance(image, (torch.Tensor, np.ndarray)):
            height, width = image.shape[-2:]
        else:
            raise ValueError("Image must be a PIL image or a tensor")

        image = v2.functional.to_image(image).to(self.device)
        image = self.transform(image).unsqueeze(0)

        state["original_height"] = height
        state["original_width"] = width
        state["backbone_out"] = self.model.backbone.forward_image(image)
        inst_interactivity_en = self.model.inst_interactive_predictor is not None
        if inst_interactivity_en and "sam2_backbone_out" in state["backbone_out"]:
            sam2_backbone_out = state["backbone_out"]["sam2_backbone_out"]
            sam2_backbone_out["backbone_fpn"][0] = (
                self.model.inst_interactive_predictor.model.sam_mask_decoder.conv_s0(
                    sam2_backbone_out["backbone_fpn"][0]
                )
            )
            sam2_backbone_out["backbone_fpn"][1] = (
                self.model.inst_interactive_predictor.model.sam_mask_decoder.conv_s1(
                    sam2_backbone_out["backbone_fpn"][1]
                )
            )
        return state

    @torch.inference_mode()
    def set_image_batch(self, images: List[np.ndarray], state=None):
        """Sets the image batch on which we want to do predictions."""
        if state is None:
            state = {}

        if not isinstance(images, list):
            raise ValueError("Images must be a list of PIL images or tensors")
        assert len(images) > 0, "Images list must not be empty"
        assert isinstance(images[0], PIL.Image.Image), (
            "Images must be a list of PIL images"
        )

        state["original_heights"] = [image.height for image in images]
        state["original_widths"] = [image.width for image in images]

        images = [
            self.transform(v2.functional.to_image(image).to(self.device))
            for image in images
        ]
        images = torch.stack(images, dim=0)
        state["backbone_out"] = self.model.backbone.forward_image(images)
        inst_interactivity_en = self.model.inst_interactive_predictor is not None
        if inst_interactivity_en and "sam2_backbone_out" in state["backbone_out"]:
            sam2_backbone_out = state["backbone_out"]["sam2_backbone_out"]
            sam2_backbone_out["backbone_fpn"][0] = (
                self.model.inst_interactive_predictor.model.sam_mask_decoder.conv_s0(
                    sam2_backbone_out["backbone_fpn"][0]
                )
            )
            sam2_backbone_out["backbone_fpn"][1] = (
                self.model.inst_interactive_predictor.model.sam_mask_decoder.conv_s1(
                    sam2_backbone_out["backbone_fpn"][1]
                )
            )
        return state
    
    @torch.inference_mode()
    def set_reference_prompt(self, ref_image, ref_box, state: Dict, 
                             ref_label: bool = True, 
                             return_heatmap: bool = True, # New Argument
                             save_path: str = None) -> Tuple[Dict, Optional[PIL.Image.Image]]:

        ref_img_tensor = v2.functional.to_image(ref_image).to(self.device)
        ref_img_tensor = self.transform(ref_img_tensor).unsqueeze(0)
        
        state_ref = self.model.backbone.forward_image(ref_img_tensor)
        ref_features = state_ref["vision_features"] # [B, C, H, W]
        
        orig_w, orig_h = ref_image.size
        x1, y1, x2, y2 = ref_box
        
        cx = (x1 + x2) / 2 / orig_w
        cy = (y1 + y2) / 2 / orig_h
        w  = (x2 - x1) / orig_w
        h  = (y2 - y1) / orig_h
        
        box_tensor = torch.tensor([cx, cy, w, h], device=self.device, dtype=torch.float32).view(1, 1, 4)
        box_mask = torch.zeros((1, 1), device=self.device, dtype=torch.bool)
        
        ref_label_value = 1 if ref_label else 0
        ref_label_tensor = torch.tensor([[ref_label_value]], device=self.device, dtype=torch.long)

        geo_prompt = Prompt(
            box_embeddings=box_tensor,
            box_mask=box_mask,
            box_labels=ref_label_tensor 
        )

        B, C, H, W = ref_features.shape
        ref_feat_seq = ref_features.flatten(2).permute(2, 0, 1)
        
        embeddings, _ = self.model.geometry_encoder(
            geo_prompt=geo_prompt,
            img_feats=[ref_feat_seq],
            img_sizes=[(H, W)]
        )
        
        if "prompt_feats" not in state:
            state["prompt_feats"] = []
            
        state["prompt_feats"].append(embeddings)
        
        if "language_features" not in state.get("backbone_out", {}):
            dummy_text = self.model.backbone.forward_text(["visual"], device=self.device)
            if "backbone_out" in state:
                state["backbone_out"].update(dummy_text)

        print(f"✅ Added {'Positive' if ref_label else 'Negative'} Exemplar Prompt.")

        heatmap_img = None
        if return_heatmap:
            if ref_features is not None:
                orig_w, orig_h = ref_image.size
                heatmap_img = self.get_activation_heatmap(ref_features, (orig_w, orig_h))
            else:
                print("⚠️ Warning: ref_features is None, cannot generate heatmap.")

        return state, heatmap_img

    @torch.inference_mode()
    def set_text_prompt(self, prompt: str, state: Dict):
        """Sets the text prompt and run the inference"""

        if "backbone_out" not in state:
            raise ValueError("You must call set_image before set_text_prompt")

        text_outputs = self.model.backbone.forward_text([prompt], device=self.device)
        # will erase the previous text prompt if any
        state["backbone_out"].update(text_outputs)
        if "geometric_prompt" not in state:
            state["geometric_prompt"] = self.model._get_dummy_prompt()

        return self._forward_grounding(state)

    @torch.inference_mode()
    def add_geometric_prompt(self, box: List, label: bool, state: Dict):
        if "backbone_out" not in state:
            raise ValueError("You must call set_image before set_text_prompt")

        if "language_features" not in state["backbone_out"]:
            # Looks like we don't have a text prompt yet. This is allowed, but we need to set the text prompt to "visual" for the model to rely only on the geometric prompt
            dummy_text_outputs = self.model.backbone.forward_text(
                ["visual"], device=self.device
            )
            state["backbone_out"].update(dummy_text_outputs)

        if "geometric_prompt" not in state:
            state["geometric_prompt"] = self.model._get_dummy_prompt()

        # adding a batch and sequence dimension
        boxes = torch.tensor(box, device=self.device, dtype=torch.float32).view(1, 1, 4)
        labels = torch.tensor([label], device=self.device, dtype=torch.bool).view(1, 1)
        state["geometric_prompt"].append_boxes(boxes, labels)

        return self._forward_grounding(state)

    

    def reset_all_prompts(self, state: Dict):
        """Removes all the prompts and results"""
        if "backbone_out" in state:
            backbone_keys_to_del = [
                "language_features",
                "language_mask",
                "language_embeds",
            ]
            for key in backbone_keys_to_del:
                if key in state["backbone_out"]:
                    del state["backbone_out"][key]

        keys_to_del = ["geometric_prompt", "boxes", "masks", "masks_logits", "scores"]
        for key in keys_to_del:
            if key in state:
                del state[key]

    @torch.inference_mode()
    def set_confidence_threshold(self, threshold: float, state=None):
        """Sets the confidence threshold for the masks"""
        self.confidence_threshold = threshold
        if state is not None and "boxes" in state:
            # we need to filter the boxes again
            # In principle we could do this more efficiently since we would only need
            # to rerun the heads. But this is simpler and not too inefficient
            return self._forward_grounding(state)
        return state

    @torch.inference_mode()
    def _forward_grounding(self, state: Dict):
        # 1. Reference(Exemplar) Embedding이 있는지 확인하고 합칩니다.
        exemplar_feats = None
        if "prompt_feats" in state and len(state["prompt_feats"]) > 0:
            # prompt_feats는 list 형태이므로 하나의 텐서로 결합합니다.
            # 예: [Tensor(1, 1, 256), Tensor(1, 1, 256)] -> Tensor(2, 1, 256)
            # geometry_encoder 출력 형태에 따라 dim=0 또는 dim=1 조정 필요
            exemplar_feats = torch.cat(state["prompt_feats"], dim=0)
            
            # (디버깅용 출력)
            # print(f"🚀 Injecting Exemplar Features: {exemplar_feats.shape}")

        # 2. 모델 Forward 실행 (exemplar_feats 인자 추가)
        outputs = self.model.forward_grounding(
            backbone_out=state["backbone_out"],
            find_input=self.find_stage,
            geometric_prompt=state["geometric_prompt"],
            find_target=None,
            exemplar_feats=exemplar_feats,  # <--- [수정됨] 모델로 전달!
        )

        out_bbox = outputs["pred_boxes"]
        out_logits = outputs["pred_logits"]
        out_masks = outputs["pred_masks"]
        
        # ... (이하 후처리 로직 기존과 동일) ...
        
        out_probs = out_logits.sigmoid()
        presence_score = outputs["presence_logit_dec"].sigmoid().unsqueeze(1)
        out_probs = (out_probs * presence_score).squeeze(-1)

        keep = out_probs > self.confidence_threshold
        out_probs = out_probs[keep]
        out_masks = out_masks[keep]
        out_bbox = out_bbox[keep]

        # convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)

        img_h = state["original_height"]
        img_w = state["original_width"]
        scale_fct = torch.tensor([img_w, img_h, img_w, img_h]).to(self.device)
        boxes = boxes * scale_fct[None, :]

        out_masks = interpolate(
            out_masks.unsqueeze(1),
            (img_h, img_w),
            mode="bilinear",
            align_corners=False,
        ).sigmoid()

        state["masks_logits"] = out_masks
        state["masks"] = out_masks > 0.5
        state["boxes"] = boxes
        state["scores"] = out_probs
        return state

    @torch.inference_mode()
    def get_activation_heatmap(self, feature_map, original_size):
        """
        feature_map: [C, H, W] tensor or numpy array
        original_size: (W, H) tuple
        Returns: PIL Image (RGB, Magma Colormap applied)
        """
        if hasattr(feature_map, 'ndim') and feature_map.ndim == 4:
            feature_map = feature_map.squeeze(0)
        
        if isinstance(feature_map, torch.Tensor):
            activation_map = torch.norm(feature_map, p=2, dim=0).detach().cpu().numpy()
        else:
            activation_map = np.linalg.norm(feature_map, axis=0)

        heatmap_min, heatmap_max = activation_map.min(), activation_map.max()
        if heatmap_max > heatmap_min:
            heatmap = (activation_map - heatmap_min) / (heatmap_max - heatmap_min)
        else:
            heatmap = activation_map

        target_w, target_h = original_size
        heatmap_resized = cv2.resize(heatmap, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        
        heatmap_rgba = plt.cm.magma(heatmap_resized)
        heatmap_rgb = (heatmap_rgba[:, :, :3] * 255).astype(np.uint8)
        
        return Image.fromarray(heatmap_rgb)