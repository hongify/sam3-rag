GOOGLE_API_KEY = "AIzaSyBcNJQXVjgj-XRqgzfxO-gxdb38y7dkuaA" 


import os
import torch
from functools import partial
from IPython.display import display, Image
from PIL import Image as PILImage 
from google import genai 
from google.genai import types

import sam3
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.agent.inference import run_single_image_inference
from sam3.agent.client_sam3 import call_sam_service as call_sam_service_orig


SAM3_ROOT = os.path.dirname(os.path.abspath(__file__))
if os.getcwd() != SAM3_ROOT:
    os.chdir(SAM3_ROOT)
print(f"Working Directory: {SAM3_ROOT}")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
torch.inference_mode().__enter__()

sam3_root = os.path.dirname(sam3.__file__)
bpe_path = f"{sam3_root}/assets/bpe_simple_vocab_16e6.txt.gz"
model = build_sam3_image_model(bpe_path=bpe_path)
processor = Sam3Processor(model, confidence_threshold=0.5)


genai.configure(api_key=GOOGLE_API_KEY)

GEMINI_MODEL_NAME = "gemini-2.5-flash" 


def send_generate_request_gemini(messages, images=None, model_name=GEMINI_MODEL_NAME, **kwargs):
    try:
        model = genai.GenerativeModel(model_name)
        content_parts = []
        if images:
            for img_path in images:
                if isinstance(img_path, str):
                    img = PILImage.open(img_path)
                    content_parts.append(img)
                else:
                    content_parts.append(img_path)


        prompt_text = ""
        if isinstance(messages, list):
            for msg in messages:
                if msg.get('role') == 'user':
                    prompt_text += msg.get('content', '') + "\n"
        elif isinstance(messages, str):
            prompt_text = messages
            
        content_parts.append(prompt_text)
        response = model.generate_content(content_parts)
        
        return response.text

    except Exception as e:
        print(f"Gemini API Error: {e}")
        return ""


image_path = "assets/cracks/GAPS384_train_0541_1_641.jpg"
image_abs_path = os.path.abspath(image_path)

prompt = "Segment and mask the crack if you can see it. If you can't, return a JSON with a 'no_mask' function call."

send_generate_request = partial(send_generate_request_gemini, model_name=GEMINI_MODEL_NAME)
call_sam_service = partial(call_sam_service_orig, sam3_processor=processor)

print("Starting inference with Gemini...")

output_image_path = run_single_image_inference(
    image_abs_path, 
    prompt, 
    {"name": GEMINI_MODEL_NAME}, 
    send_generate_request, 
    call_sam_service,
    debug=True, 
    output_dir="agent_output"
)



if output_image_path is not None:
    print(f"Output saved to: {output_image_path}")
    display(Image(filename=output_image_path))
else:
    print("Inference finished, but no output image path returned.")