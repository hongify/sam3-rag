import os
import torch
from functools import partial
from IPython.display import display, Image

import sam3
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.agent.client_llm import send_generate_request as send_generate_request_orig
from sam3.agent.client_sam3 import call_sam_service as call_sam_service_orig
from sam3.agent.inference import run_single_image_inference

# 작업 디렉토리 설정
SAM3_ROOT = os.path.dirname(os.path.abspath(__file__))
if os.getcwd() != SAM3_ROOT:
    os.chdir(SAM3_ROOT)
print(f"Working Directory: {SAM3_ROOT}")

# GPU 및 PyTorch 설정
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
torch.inference_mode().__enter__()

# SAM3 모델 및 프로세서 로드
sam3_root = os.path.dirname(sam3.__file__)
bpe_path = f"{sam3_root}/assets/bpe_simple_vocab_16e6.txt.gz"
model = build_sam3_image_model(bpe_path=bpe_path)
processor = Sam3Processor(model, confidence_threshold=0.5)

# ---------------------------------------------------------
# OpenAI API 설정 (이 부분만 수정하세요)
# ---------------------------------------------------------
OPENAI_API_KEY = "AIzaSyBcNJQXVjgj-XRqgzfxO-gxdb38y7dkuaA"
OPENAI_MODEL = "gemini-2.5-flash" # gpt-4-turbo 등 다른 모델로 변경 가능

LLM_SERVER_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

llm_config = {
    "provider": "GEMINI_API_KEY",
    "model": OPENAI_MODEL,
    "name": OPENAI_MODEL,
    "api_key": OPENAI_API_KEY
}
# ---------------------------------------------------------

# 추론용 이미지 및 프롬프트 설정
image_path = "assets/cracks/GAPS384_train_0541_1_641.jpg"
image_abs_path = os.path.abspath(image_path)

prompt = "segment and mask the crack if you can see it. if you can't return no mask function in json"

# Partial 함수 생성 (OpenAI 서버로 요청을 보내도록 설정)
send_generate_request = partial(
    send_generate_request_orig, 
    server_url=LLM_SERVER_URL, 
    model=llm_config["model"], 
    api_key=llm_config["api_key"]
)
call_sam_service = partial(call_sam_service_orig, sam3_processor=processor)

print(f"Starting inference with OpenAI ({OPENAI_MODEL})...")

# 추론 실행
output_image_path = run_single_image_inference(
    image_abs_path, 
    prompt, 
    llm_config, 
    send_generate_request, 
    call_sam_service,
    debug=True, 
    output_dir="agent_output"
)

# 결과 출력
if output_image_path is not None:
    print(f"Output saved to: {output_image_path}")
    display(Image(filename=output_image_path))
else:
    print("Inference finished, but no output image path returned.")