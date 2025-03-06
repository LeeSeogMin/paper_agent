"""
Test script to check if the web application is running
"""

import requests
import sys

def test_web_app():
    """Test if the web application is running"""
    try:
        # Try to access the home page
        response = requests.get('http://localhost:8080/')
        if response.status_code == 200:
            print("웹 애플리케이션이 실행 중입니다! 상태 코드:", response.status_code)
            return True
        else:
            print("웹 애플리케이션이 오류를 반환했습니다. 상태 코드:", response.status_code)
            return False
    except requests.exceptions.ConnectionError:
        print("웹 애플리케이션에 연결할 수 없습니다. http://localhost:8080/에서 실행 중인지 확인하세요.")
        return False

if __name__ == '__main__':
    # Test the web application
    app_running = test_web_app()
    
    if app_running:
        print("\n성공! 웹 애플리케이션이 실행 중입니다.")
        print("다음 주소에서 접속할 수 있습니다: http://localhost:8080/")
        sys.exit(0)
    else:
        print("\n웹 애플리케이션이 실행되고 있지 않습니다.")
        sys.exit(1) 