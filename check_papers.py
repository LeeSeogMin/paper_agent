import json
import os
import traceback

def check_papers_json():
    """papers.json 파일을 확인하는 함수"""
    file_path = 'data/papers.json'
    
    # 파일 존재 확인
    if not os.path.exists(file_path):
        print(f"파일이 존재하지 않습니다: {file_path}")
        return
    
    # 파일 크기 확인
    file_size = os.path.getsize(file_path)
    print(f"파일 크기: {file_size} 바이트")
    
    try:
        # 파일 내용 읽기
        with open(file_path, 'r', encoding='utf-8') as f:
            papers = json.load(f)
        
        # 기본 정보 출력
        print(f"파일에 {len(papers)}개의 논문 정보가 있습니다.")
        
        # 첫 번째 논문 정보 출력
        if papers and len(papers) > 0:
            first_paper = papers[0]
            print("\n첫 번째 논문 정보:")
            print(f"ID: {first_paper.get('id', '정보 없음')}")
            print(f"제목: {first_paper.get('title', '정보 없음')}")
            
            # 'topic modeling' 키워드 검색
            keyword = "topic modeling"
            keywords = keyword.lower().split()
            
            # 검색 결과 저장
            search_results = []
            
            for paper in papers:
                title = paper.get('title', '').lower()
                abstract = paper.get('abstract', '').lower()
                
                # 간단한 관련성 점수 계산 (키워드 일치 수)
                relevance = sum(1 for kw in keywords if kw in title or kw in abstract)
                
                if relevance > 0:
                    paper['relevance'] = relevance
                    search_results.append(paper)
            
            # 검색 결과 출력
            print(f"\n'{keyword}' 키워드로 검색한 결과: {len(search_results)}개 논문 찾음")
            
    except Exception as e:
        print(f"오류 발생: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    check_papers_json() 