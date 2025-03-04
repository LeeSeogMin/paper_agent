# 요약 체인 초기화
self.summary_chain = LLMChain(
    llm=self.llm,
    prompt=SUMMARY_PROMPT,
    verbose=self.verbose
)

# 키워드 추출 체인 초기화
self.keyword_chain = LLMChain(
    llm=self.llm,
    prompt=KEYWORD_EXTRACTION_PROMPT,
    verbose=self.verbose
) 