import os
import sys
import io
import uuid
import base64
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
from PIL import Image

# 상위 디렉토리에서 모듈 import 가능하도록 설정
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from generator import generate_redesign
from prompt_suggester import generate_caption, suggest_prompt_langchain, translate_to_english
from utils import load_yaml

# 설정 로드
config = load_yaml("../config.yaml")
app = FastAPI()

# ✅ 프롬프트 생성 엔드포인트 (프리셋 포함)
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


# ✅ 리디자인 이미지 생성
@app.post("/redesign_image")
async def redesign_image(prompt: str = Form(...), file: UploadFile = None):
    """
    업로드된 이미지와 프롬프트를 이용하여 리디자인된 이미지를 생성하여 base64로 반환
    """
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        output_image = generate_redesign(image, prompt)

        # 결과 이미지를 base64로 인코딩하여 전달
        buffer = io.BytesIO()
        output_image.save(buffer, format="PNG")
        encoded_image = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return JSONResponse({"image_base64": encoded_image})
    except Exception as e:
        print("[Redesign Error]", e)
        return JSONResponse({"error": str(e)}, status_code=500)
