from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import os
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import SystemMessage, HumanMessage

# OpenAI API Key 설정


# 1. BLIP 설정 (이미지 → 설명)
device = "cuda" if torch.cuda.is_available() else "cpu"
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

def generate_caption(image: Image.Image) -> str:
    """
    이미지에 대한 캡션을 생성 (BLIP 기반)
    """
    inputs = blip_processor(image, return_tensors="pt").to(device)
    out = blip_model.generate(**inputs, max_new_tokens=30)
    caption = blip_processor.decode(out[0], skip_special_tokens=True)
    return caption


# 2. LangChain 기반 프롬프트 생성 함수
def suggest_prompt_langchain(caption: str) -> str:
    """
    LangChain PromptTemplate을 사용한 GPT-3.5 기반 리디자인 명령 생성
    """
    # LangChain LLM 설정
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0.7,
        max_tokens=50
    )

    # 템플릿 정의
    prompt_template = PromptTemplate(
        input_variables=["caption"],
        template=(
            "이 설명은 옷 이미지에 대한 것입니다:\n\"{caption}\"\n\n"
            "사용자가 이 옷을 어떻게 리디자인하면 좋을지 한국어로 1문장으로 추천해 주세요. "
            "예시: \"please change this cloth to red color\", \"please remove the logo\"\n\n"
            "추천:"
        )
    )

    prompt = prompt_template.format(caption=caption)

    try:
        response = llm([
            SystemMessage(content="당신은 패션 리디자인 전문가입니다."),
            HumanMessage(content=prompt)
        ])
        return response.content.strip()
    except Exception as e:
        print("[LangChain GPT Error]", e)
        return "이 옷을 더 스타일리시하게 바꿔줘"
    
def translate_to_english(korean_prompt: str) -> str:
    """
    한국어 프롬프트를 고품질 영어 프롬프트로 변환
    (Stable Diffusion용 프리셋 스타일 구성: 의류 변경 내용 + 스타일 + 품질 + 구도)
    """
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0.5,
        max_tokens=200
    )

    try:
        response = llm([
            SystemMessage(
                content=(
                    "You are a professional prompt engineer for Stable Diffusion XL. "
                    "Your job is to convert Korean fashion redesign instructions into high-quality English prompts "
                    "suitable for text-to-image generation.\n\n"
                    "💡 Important Instructions:\n"
                    "- Do NOT write full sentences.\n"
                    "- Output should be a list of descriptive **keywords**.\n"
                    "- Use a **preset-style structure** combining:\n"
                    "   1. Clothing description (translated and redesigned)\n"
                    "   2. Style type (e.g., high fashion, editorial photo)\n"
                    "   3. Viewpoint (e.g., full body, front view)\n"
                    "   4. Lighting/quality (e.g., soft lighting, 8k, studio background)\n\n"
                    "🎯 Final Output Format Example:\n"
                    "   blue sleeveless hoodie, high fashion, full body, 8k, soft lighting, studio background"
                )
            ),
            HumanMessage(
                content=f'Korean Instruction: "{korean_prompt}"\n\nPlease return the full prompt below:'
            )
        ])
        return response.content.strip()
    except Exception as e:
        print("[Prompt Translation Error]", e)
        return "blue sleeveless hoodie, high fashion, full body, 8k, soft lighting, studio background"
    