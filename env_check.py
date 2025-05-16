from dotenv import load_dotenv
import os
# API KEY 정보로드
load_dotenv()
print(f"[API KEY]\n{os.environ['OPENAI_API_KEY']}")