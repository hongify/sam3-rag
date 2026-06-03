import os
SAM3_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(SAM3_ROOT)
print(SAM3_ROOT)

import torch
from functools import partial
from IPython.display import display, Image

import sam3
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.agent.client_llm import send_generate_request as send_generate_request_orig
from sam3.agent.client_sam3 import call_sam_service as call_sam_service_orig
from sam3.agent.inference import run_single_image_inference

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# _ = os.system("nvidia-smi")

LLM_CONFIGS = {
    # vLLM-served models
    "qwen3_vl_4b_thinking-fp8": {
        "provider": "vllm",
        "model": "Qwen/Qwen3-VL-4B-Thinking-FP8",
    },
    "qwen3_vl_8b_thinking": {
        "provider": "vllm",
        "model": "Qwen/Qwen3-VL-8B-Thinking",
    },
    
} 


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
torch.inference_mode().__enter__()




sam3_root = os.path.dirname(sam3.__file__)
bpe_path = f"{sam3_root}/assets/bpe_simple_vocab_16e6.txt.gz"
model = build_sam3_image_model(bpe_path=bpe_path)
processor = Sam3Processor(model, confidence_threshold=0.5)


model = "qwen3_vl_4b_thinking-fp8"
LLM_API_KEY = "DUMMY_API_KEY"

llm_config = LLM_CONFIGS[model]
llm_config["api_key"] = LLM_API_KEY
llm_config["name"] = model

if llm_config["provider"] == "vllm":
    LLM_SERVER_URL = "https://powerseller-motels-qualifications-technological.trycloudflare.com/v1"  # replace this with your vLLM server address as needed
else:
    LLM_SERVER_URL = llm_config["base_url"]



image = "assets/cracks/GAPS384_train_0541_1_641.jpg"
prompt = "segment and mask the crack if you can see it. if you can't return no mask function in json"



image = os.path.abspath(image)
send_generate_request = partial(send_generate_request_orig, server_url=LLM_SERVER_URL, model=llm_config["model"], api_key=llm_config["api_key"])
call_sam_service = partial(call_sam_service_orig, sam3_processor=processor)
output_image_path = run_single_image_inference(
    image, prompt, llm_config, send_generate_request, call_sam_service,
    debug=True, output_dir="agent_output"
)

if output_image_path is not None:
    display(Image(filename=output_image_path))



    