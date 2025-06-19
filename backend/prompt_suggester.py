from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import os
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import SystemMessage, HumanMessage
from transformers import CLIPTokenizer
from typing import List

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
                    "   Clothing description (translated and redesigned),Style type (e.g., high fashion, editorial photo)"
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

def truncate_prompt(prompt: str, max_tokens: int = 77) -> str:
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    tokens = tokenizer.tokenize(prompt)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
    return tokenizer.convert_tokens_to_string(tokens)

def improve_prompt_with_llm(bad_prompt: str, visual_goals: List[str] = None) -> str:
    """
    기존 프롬프트를 명확하게 시각적으로 보완하는 함수
    - bad_prompt: 원래의 부정확하거나 모호한 프롬프트
    - visual_goals: 프롬프트에 반드시 반영되어야 할 시각적 목표들
    """
    if visual_goals is None:
        visual_goals = [
            "Place a large white logo at the center of the chest",
            "Use a red sweatshirt as the base garment",
            "Ensure the logo is bold and clearly visible",
        ]

    goal_text = "\n".join(f"- {g}" for g in visual_goals)

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.5)

    messages = [
        SystemMessage(content="You are a professional prompt engineer for Stable Diffusion. Your job is to rewrite prompts to be more visually descriptive and specific."),
        HumanMessage(content=f"""
Original Prompt:
\"{bad_prompt}\"

Visual Goals to Apply:
{goal_text}

Please rewrite the prompt in a visually descriptive and effective format for Stable Diffusion. Keep it concise but specific.
""")
    ]

    improved_prompt = llm(messages).content.strip()
    return improved_prompt