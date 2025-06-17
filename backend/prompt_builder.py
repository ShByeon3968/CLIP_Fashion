'''
user input text를 받아서 image redesign에 사용할 프롬포트 생성
'''
def build_prompt(user_input: str) -> str:
    # 예시: "파란색으로 바꿔줘" → "a blue dress, best quality, studio lighting"
    color_map = {
        "파란색": "blue",
        "빨간색": "red",
        "흰색": "white",
        "검은색": "black"
    }
    prompt = "a stylish fashion photo, "
    for kr, en in color_map.items():
        if kr in user_input:
            prompt += f"{en} clothes, "
    return prompt + "fashion magazine style, clean background"
