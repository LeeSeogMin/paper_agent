from utils.xai_client import XAIClient  # utils 폴더에서 가져오기

def _edit_content(self, content, instructions, style_guide=None):
    """XAI API를 사용하여 콘텐츠 편집"""
    
    # 싱글톤 인스턴스 가져오기
    xai_client = XAIClient.get_instance()
    
    system_prompt = "You are a professional editor."
    if style_guide:
        system_prompt += f" Follow this style guide: {style_guide}"
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Edit the following content according to these instructions: {instructions}\n\nContent: {content}"}
    ]
    
    try:
        response = xai_client.chat_completion(messages=messages)
        return response.get("choices", [{}])[0].get("message", {}).get("content", "")
    except Exception as e:
        self.logger.error(f"XAI API 호출 오류: {str(e)}")
        return f"콘텐츠 편집 중 오류 발생: {str(e)}" 