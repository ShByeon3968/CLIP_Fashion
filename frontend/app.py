import gradio as gr
from search_engine import search_by_text
from app_utils import load_yaml
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
        return Image.open(io.BytesIO(decoded)), clip_score, prompt, Image.open(io.BytesIO(decoded))
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
        return Image.open(io.BytesIO(decoded_imp)), clip_score_imp, prompt_imp, Image.open(io.BytesIO(decoded_imp))
    else:
        raise RuntimeError(f"API 응답 실패: {res.status_code} {res.text}")
    
def request_try_on(user_img_dict, garm_img, garment_des, is_checked, is_checked_crop, denoise_steps, seed):
    try:
        # 사용자 이미지 변환
        print(user_img_dict)
        user_img = user_img_dict["background"] if isinstance(user_img_dict, dict) else user_img_dict
        user_buf = io.BytesIO()
        user_img.save(user_buf, format="PNG")
        user_buf.seek(0)

        # 의류 이미지 변환
        garm_buf = io.BytesIO()
        garm_img.save(garm_buf, format="PNG")
        garm_buf.seek(0)

        files = {
            "user_image": ("user.png", user_buf, "image/png"),
            "garment_image": ("garment.png", garm_buf, "image/png")
        }

        data = {
            "garment_des": garment_des,
            "use_mask": str(is_checked).lower(),       # 'true' or 'false'
            "use_crop": str(is_checked_crop).lower(),  # 'true' or 'false'
            "denoise_steps": int(denoise_steps),
            "seed": int(seed)
        }

        res = requests.post(f"{API_URL}/virtual_try_on/", files=files, data=data)

        if res.status_code != 200:
            raise RuntimeError(f"API 응답 실패: {res.status_code} {res.text}")

        json_data = res.json()
        tryon_base64 = json_data["result"]
        masked_base64 = json_data.get("masked", "")

        tryon_image = Image.open(io.BytesIO(base64.b64decode(tryon_base64)))
        masked_image = Image.open(io.BytesIO(base64.b64decode(masked_base64))) if masked_base64 else None

        return tryon_image, masked_image

    except Exception as e:
        raise RuntimeError(f"Try-on 요청 실패: {str(e)}")
    
# Gradio UI
with gr.Blocks() as app:
    gr.Markdown("## 👕 텍스트 검색 기반 의류 리디자인 시스템")

    selected_image_path = gr.State()
    generated_prompt = gr.State()
    
    # TAB 1: 의류 이미지 선택 (텍스트 검색 or 직접 업로드)
    with gr.Tab("1️⃣ 의류 이미지 선택"):
        gr.Markdown("### 🔍 텍스트 검색 또는 📁 직접 업로드로 의류 이미지를 선택하세요")

        image_input_mode = gr.Radio(
            label="이미지 선택 방식",
            choices=["텍스트로 유사 이미지 검색", "이미지 직접 업로드"],
            value="텍스트로 유사 이미지 검색"
        )

        # 1-1. 텍스트 검색 UI
        with gr.Column(visible=True) as text_search_area:
            text_input = gr.Textbox(label="예: 검은 후드티")
            search_button = gr.Button("검색")
            result_gallery = gr.Gallery(label="유사 이미지 선택", show_label=False)

            search_button.click(fn=find_by_text, inputs=text_input, outputs=result_gallery)
            result_gallery.select(fn=select_image, inputs=[result_gallery], outputs=selected_image_path)

    # 1-2. 직접 업로드 UI
    with gr.Column(visible=False) as upload_area:
        upload_image = gr.Image(type="filepath", label="직접 의류 이미지 업로드")
        confirm_button = gr.Button("이 이미지 선택")
        confirm_button.click(fn=lambda path: path, inputs=upload_image, outputs=selected_image_path)

    # Radio 버튼에 따라 보이기 전환
    def toggle_mode(mode):
        return (
            gr.update(visible=(mode == "텍스트로 유사 이미지 검색")),
            gr.update(visible=(mode == "이미지 직접 업로드"))
        )

    image_input_mode.change(fn=toggle_mode, inputs=image_input_mode, outputs=[text_search_area, upload_area])

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
        redesign_result = gr.Image(label="리디자인 결과",scale=0.5)
        clip_score_text = gr.Textbox(label="CLIP Score", interactive=False)
        apply_prompt_button = gr.Button("💡 보완 프롬프트 적용", visible=False)

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

    with gr.Tab("4️⃣ 시착 이미지 생성"):
        with gr.Column():
            imgs = gr.ImageEditor(sources='upload', type="pil", label='사용자 인물 이미지 (자동 마스킹 가능)', interactive=True)
            with gr.Row():
                is_checked = gr.Checkbox(label="자동 마스크 사용", value=True)
            with gr.Row():
                is_checked_crop = gr.Checkbox(label="자동 자르기", value=False)

        with gr.Column():
            garm_img = gr.Image(label="의류 이미지", sources='upload', type="pil")
            with gr.Row(elem_id="prompt-container"):
                with gr.Row():
                    prompt = gr.Textbox(placeholder="예: Short Sleeve Round Neck T-shirts", show_label=True)

        with gr.Column():
            masked_img = gr.Image(label="마스킹된 사용자 이미지", show_share_button=False,scale=0.5)
        with gr.Column():
            image_out = gr.Image(label="Try-on 결과 이미지", show_share_button=False,scale=0.5)
        
        with gr.Column():
            try_button = gr.Button(value="Try-on 실행")
            denoise_steps = gr.Number(label="Denoising Steps", minimum=20, maximum=40, value=30, step=1)
            seed = gr.Number(label="Seed", minimum=-1, maximum=2147483647, step=1, value=42)
        

        
        # 리디자인 생성 + CLIP Score 반영 + 보완 프롬프트 저장
        redesign_button.click(
            fn=request_redesign,
            inputs=[design_prompt, selected_image_path],
            outputs=[redesign_result, clip_score_text, design_prompt,garm_img]  
        )
        apply_prompt_button.click(
            fn=requset_redesign_with_improved_prompt,
            inputs=[design_prompt, selected_image_path],
            outputs=[redesign_result, clip_score_text, design_prompt,garm_img]  # design_prompt가 보완되면 자동 반영
        )

        try_button.click(fn=request_try_on, inputs=[imgs, garm_img, prompt, is_checked,is_checked_crop, denoise_steps, seed], outputs=[image_out,masked_img], api_name='tryon')

# 앱 실행
app.launch()
