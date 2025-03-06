"""
텍스트 처리 유틸리티 함수들을 제공하는 모듈입니다.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity

def cosine_similarity(text1, text2):
    """
    두 텍스트 간의 코사인 유사도를 계산합니다.
    
    Args:
        text1 (str): 첫 번째 텍스트
        text2 (str): 두 번째 텍스트
        
    Returns:
        float: 코사인 유사도 (0~1 사이의 값, 1이 가장 유사함)
    """
    if not text1 or not text2:
        return 0.0
        
    try:
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([text1, text2])
        similarity = sklearn_cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return float(similarity)
    except Exception as e:
        print(f"코사인 유사도 계산 중 오류 발생: {str(e)}")
        return 0.0 