import gradio as gr
from search_engine import search_by_text
from utils import load_yaml
from PIL import Image
import requests
import io
import base64

# 환경설정 로딩
config = load_yaml("config.yaml")
API_URL = config["API_URL"]

# 검색 함수
def find_by_text(text):
    result_paths = search_by_text(text)
    return result_paths

# 갤러리 이미지 선택 시 경로 저장
def select_image(evt: gr.SelectData, all_images):
    selected_path = all_images[evt.index][0]
    print("선택된 이미지:", selected_path)
    return selected_path

# 프롬프트 생성 API 요청
def request_prompt_generation(image_path, style, angle, lighting):
    print("요청 이미지:", image_path)
    res = requests.post(
        f"{API_URL}/generate_prompt",
        data={
            "text": image_path,
            "style": style,
            "angle": angle,
            "lighting": lighting
        }
    )
    json = res.json()
    if "ko_prompt" not in json or "en_prompt" not in json:
        raise RuntimeError(f"API 오류: {json.get('error', '응답 형식 이상')}")
    return json["ko_prompt"], json["en_prompt"]

# 리디자인 요청 API
def request_redesign(prompt, image_path):
    image = Image.open(image_path).convert("RGB")
    img_bytes = io.BytesIO()
    image.save(img_bytes, format="PNG")
    img_bytes.seek(0)

    files = {"file": ("image.png", img_bytes, "image/png")}
    data = {"prompt": prompt}

    res = requests.post(f"{API_URL}/redesign_image", files=files, data=data)

    if res.status_code == 200:
        json_data = res.json()
        image_base64 = json_data["image_base64"]
        clip_score = json_data.get("clip_score", 0.0)
        prompt = json_data.get("prompt", "")
        decoded = base64.b64decode(image_base64)
        return Image.open(io.BytesIO(decoded)), clip_score, prompt
    else:
        raise RuntimeError(f"API 응답 실패: {res.status_code} {res.text}")
    
def requset_redesign_with_improved_prompt(pre_prompt,image_path):
    image = Image.open(image_path).convert("RGB")
    img_bytes = io.BytesIO()
    image.save(img_bytes, format="PNG")
    img_bytes.seek(0)

    files = {"file": ("image.png", img_bytes, "image/png")}
    data = {"pre_prompt": pre_prompt}

    res = requests.post(f"{API_URL}/redesign_improved", files=files,data=data)

    if res.status_code == 200:
        json_data = res.json()
        image_base64_imp = json_data["image_base64"]
        clip_score_imp = json_data.get("clip_score", 0.0)
        prompt_imp = json_data.get("prompt", "")
        decoded_imp = base64.b64decode(image_base64_imp)
        return Image.open(io.BytesIO(decoded_imp)), clip_score_imp, prompt_imp
    else:
        raise RuntimeError(f"API 응답 실패: {res.status_code} {res.text}")

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## 👕 텍스트 검색 기반 의류 리디자인 시스템")

    selected_image_path = gr.State()
    generated_prompt = gr.State()

    # TAB 1: 텍스트로 유사 이미지 검색
    with gr.Tab("1️⃣ 텍스트로 유사 이미지 검색"):
        text_input = gr.Textbox(label="예: 검은 후드티")
        search_button = gr.Button("검색")
        result_gallery = gr.Gallery(label="유사 이미지 선택", show_label=False)

        search_button.click(fn=find_by_text, inputs=text_input, outputs=result_gallery)
        result_gallery.select(fn=select_image, inputs=[result_gallery], outputs=selected_image_path)

    # TAB 2: 프롬프트 자동 생성 + 프리셋 UI
    with gr.Tab("2️⃣ 프롬프트 자동 생성"):
        prompt_ko_display = gr.Textbox(label="추천 프롬프트 (한국어)")
        prompt_en_display = gr.Textbox(label="영어 Prompt (SDXL용)", lines=2,interactive=True)
        generate_prompt_button = gr.Button("자동 프롬프트 생성")

        # ✅ 프리셋 선택 항목
        style_dropdown = gr.Dropdown(
            label="스타일 프리셋",
            choices=["high fashion", "streetwear", "vintage", "modern minimalist"],
            value="high fashion"
        )
        angle_dropdown = gr.Dropdown(
            label="카메라 앵글",
            choices=["full body", "waist up", "close-up", "side view"],
            value="full body"
        )
        light_dropdown = gr.Dropdown(
            label="조명 및 배경",
            choices=["studio background, soft lighting, 8k", "outdoor sunlight", "runway lighting"],
            value="studio background, soft lighting, 8k"
        )

        generate_prompt_button.click(
            fn=request_prompt_generation,
            inputs=[selected_image_path, style_dropdown, angle_dropdown, light_dropdown],
            outputs=[prompt_ko_display, prompt_en_display]
        )

    # TAB 3: 리디자인 요청
    with gr.Tab("3️⃣ 리디자인 요청"):
        design_prompt = gr.Textbox(label="리디자인 지시 (자동 입력 가능)")
        redesign_button = gr.Button("리디자인 생성")
        redesign_result = gr.Image(label="리디자인 결과")
        clip_score_text = gr.Textbox(label="CLIP Score", interactive=False)
        apply_prompt_button = gr.Button("💡 보완 프롬프트 적용", visible=False)

        # 리디자인 생성 + CLIP Score 반영 + 보완 프롬프트 저장
        redesign_button.click(
            fn=request_redesign,
            inputs=[design_prompt, selected_image_path],
            outputs=[redesign_result, clip_score_text, design_prompt]  
        )

        # 보완 프롬프트 버튼 제어 로직
        def check_clip_score(score):
            try:
                score = float(score)
                return gr.update(visible=(score <= 0.24))
            except:
                return gr.update(visible=False)

        clip_score_text.change(
            fn=check_clip_score,
            inputs=clip_score_text,
            outputs=apply_prompt_button
        )

        apply_prompt_button.click(
            fn=requset_redesign_with_improved_prompt,
            inputs=[design_prompt, selected_image_path],
            outputs=[redesign_result, clip_score_text, design_prompt]  # design_prompt가 보완되면 자동 반영
        )

# 앱 실행
demo.launch()
