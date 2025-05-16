from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage

class TextParser:
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0
        )
    
    def extract_action_info(self,prompt:str) -> dict:
        """
        자연어로 입력된 문장을 받아서 행동(action), 스타일(style), 카테고리(category) 등의 구조화된 정보를 추출
        Args:
            prompt (str): 자연어로 입력된 문장
        Returns:
            dict: 구조화된 정보
        """
        system_msg = (
        "You're a parser that extracts structured animation commands "
        "from natural language describing human motion. Extract 'subject', 'action', 'style', and 'scene'."            
        )
        user_prompt = f"""
        Text: "{prompt}"
        Please return as JSON like:
        {{
            "subject": "...",
            "action": "...",
            "style": "...",
            "scene": "..."
        }}
        """

        response = self.llm([HumanMessage(content=system_msg), HumanMessage(content=user_prompt)])
        try:
            json_result = eval(response.content.strip())  # 안전하게 바꾸려면 json.loads 등으로 교체
            return json_result
        except:
            return {"error": "Parsing failed", "raw_output": response.content}



