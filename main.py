from text_parser.parse_text import TextParser
from animation_diff.generate_frames import AnimateDiff

if __name__ == "__main__":
    # Initialize the TextParser
    text_parser = TextParser()
    # Initialize the AnimateDiff
    animate_diff = AnimateDiff()

    # Example input
    text = "A man in a suit enters a room and speaks to two women sitting on a couch. The man, wearing a dark suit with a gold tie, enters the room from the left and walks towards the center of the frame. He has short gray hair, light skin, and a serious expression. He places his right hand on the back of a chair as he approaches the couch. Two women are seated on a light-colored couch in the background. The woman on the left wears a light blue sweater and has short blonde hair. The woman on the right wears a white sweater and has short blonde hair. The camera remains stationary, focusing on the man as he enters the room. The room is brightly lit, with warm tones reflecting off the walls and furniture. The scene appears to be from a film or television show."

    # 1단계 자연어 분석
    parsed = text_parser.extract_action_info(text)
    if "error" in parsed:
        print("Parsing failed:", parsed)
        exit(1)

    print("Parsed Prompt:", parsed)

    # 2단계: 텍스트 재구성
    prompt = f"A {parsed['subject']} is {parsed['action']} {parsed['style']} {parsed['scene']}"
    print("Final Prompt for Animation:", prompt)

    # 3단계: 애니메이션 생성
    result_dir = animate_diff.generate_animation(prompt=text, num_frames=160)
    print(f"Animation frames saved in: {result_dir}")