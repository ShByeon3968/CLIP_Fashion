import os
import sys
import io
import base64
from fastapi import FastAPI, UploadFile, Form, File
from fastapi.responses import JSONResponse
from PIL import Image

# 상위 디렉토리에서 모듈 import 가능하도록 설정
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from generator import generate_redesign, generate_redesign_img2img
from prompt_suggester import generate_caption, suggest_prompt_langchain, translate_to_english, improve_prompt_with_llm
from app_utils import load_yaml, pil_to_base64
from clip_score_util import compute_clip_score
from ldm_vton import start_tryon

# 설정 로드
config = load_yaml("../config.yaml")
app = FastAPI()

# 프롬프트 생성 엔드포인트 (프리셋 포함)
@app.post("/generate_prompt")
async def generate_prompt(
    text: str = Form(...),  # 이미지 경로
    style: str = Form(...),
    angle: str = Form(...),
    lighting: str = Form(...)
):
    """
    선택된 이미지 → 캡션 → 한국어 프롬프트 → 영어 프롬프트 + 프리셋
    """
    try:
        # 이미지 열기
        target_image = Image.open(text).convert("RGB")

        # BLIP 기반 캡션 생성
        caption = generate_caption(target_image)
        print(f"[Caption] {caption}")

        # 한국어 리디자인 프롬프트 생성
        ko_prompt = suggest_prompt_langchain(caption)
        print(f"[Korean Prompt] {ko_prompt}")

        # 영어 번역 + 프리셋 후처리
        en_prompt_base = translate_to_english(ko_prompt)
        print(f"[English Prompt (before preset)] {en_prompt_base}")

        preset_suffix = f"{style}, {angle}, {lighting}"
        en_prompt_full = f"{en_prompt_base}, {preset_suffix}"

        return {
            "ko_prompt": ko_prompt,
            "en_prompt": en_prompt_full
        }

    except Exception as e:
        print("[Error]", e)
        return JSONResponse({"error": str(e)}, status_code=500)


# 리디자인 이미지 생성 + 프롬프트 보완 루프
@app.post("/redesign_image")
async def redesign_image(prompt: str = Form(...), file: UploadFile = None):
    try:
        contents = await file.read()
        input_image = Image.open(io.BytesIO(contents)).convert("RGB")

        output_image = generate_redesign_img2img(input_image, prompt)
        clip_score = compute_clip_score(output_image, prompt)
        print(f"[Initial CLIPScore] {clip_score}")

        buffer = io.BytesIO()
        output_image.save(buffer, format="PNG")
        encoded_image = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return JSONResponse({
            "image_base64": encoded_image,
            "prompt": prompt,
            "clip_score": clip_score
        })

    except Exception as e:
        print("[Redesign Error]", e)
        return JSONResponse({"error": str(e)}, status_code=500)
    
CLIP_THRESHOLD = 0.24
@app.post("/redesign_improved")
async def redesign_improved(pre_prompt: str = Form(...),file: UploadFile = None):
    try:
        contents = await file.read()
        input_image = Image.open(io.BytesIO(contents)).convert("RGB")

        improved_prompt = improve_prompt_with_llm(pre_prompt)
        output_image = generate_redesign_img2img(input_image, improved_prompt)
        clip_score = compute_clip_score(output_image, improved_prompt)
        print(f"[Improved CLIPScore] {clip_score}")

        buffer = io.BytesIO()
        output_image.save(buffer, format="PNG")
        encoded_image = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return JSONResponse({
            "image_base64": encoded_image,
            "prompt": improved_prompt,
            "clip_score": clip_score
        })

    except Exception as e:
        print("[Redesign Improved Error]", e)
        return JSONResponse({"error": str(e)}, status_code=500)
    
@app.post("/virtual_try_on/")
async def virtual_tryon(
    user_image: UploadFile = File(...),
    garment_image: UploadFile = File(...),
    garment_des: str = Form(...),
    use_mask: str = Form("true"),
    use_crop: str = Form("false"),
    denoise_steps: int = Form(30),
    seed: int = Form(42)
):
    try:
        # 이미지 로드
        user_bytes = await user_image.read()
        garment_bytes = await garment_image.read()

        user_img = Image.open(io.BytesIO(user_bytes)).convert("RGB")
        garment_img = Image.open(io.BytesIO(garment_bytes)).convert("RGB")
        garment_des = garment_des.strip()
        is_checked = use_mask.lower()
        is_checked_crop = use_crop.lower()

        image_out,masked_img = start_tryon(user_img,garment_img,garment_des,is_checked,is_checked_crop,denoise_steps,seed)

        return JSONResponse({
            "result": pil_to_base64(image_out),
            "masked": pil_to_base64(masked_img)
        })
    
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)      
