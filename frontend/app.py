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
def request_prompt_generation(user_input):
    print("사용자 입력:", user_input)
    res = requests.post(f"{API_URL}/generate_prompt", data={"text": user_input})
    return res.json()["ko_prompt"], res.json()["en_prompt"]

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
        image_base64 = res.json()["image_base64"]
        decoded = base64.b64decode(image_base64)
        return Image.open(io.BytesIO(decoded))
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

    # TAB 2: 프롬프트 자동 생성
    with gr.Tab("2️⃣ 프롬프트 자동 생성"):
        prompt_ko_display = gr.Textbox(label="추천 프롬프트 (한국어)")
        prompt_en_display = gr.Textbox(label="영어 Prompt (SDXL용)", lines=2)
        generate_prompt_button = gr.Button("자동 프롬프트 생성")

        generate_prompt_button.click(
            fn=request_prompt_generation,
            inputs=selected_image_path,
            outputs=[prompt_ko_display, prompt_en_display]
        )

    # TAB 3: 리디자인 요청
    with gr.Tab("3️⃣ 리디자인 요청"):
        design_prompt = gr.Textbox(label="리디자인 지시 (자동 입력 가능)")
        redesign_button = gr.Button("리디자인 생성")
        redesign_result = gr.Image(label="리디자인 결과")

        # design_prompt가 선언된 이후
        generate_prompt_button.click(
            fn=lambda ko, en: (en, en),
            inputs=[prompt_ko_display, prompt_en_display],
            outputs=[generated_prompt, design_prompt]
        )

        redesign_button.click(
            fn=request_redesign,
            inputs=[design_prompt, selected_image_path],
            outputs=redesign_result
        )

# 앱 실행
demo.launch()
