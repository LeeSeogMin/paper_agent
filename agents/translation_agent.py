# 번역 체인 초기화
self.translation_chain = LLMChain(
    llm=self.llm,
    prompt=TRANSLATION_PROMPT,
    verbose=self.verbose
) 