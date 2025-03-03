# -*- coding: utf-8 -*-

"""
프로젝트 내 모든 Python 파일에서 null 바이트를 제거하는 스크립트
"""

import os
import fnmatch

def remove_null_bytes(directory, patterns=['*.py']):
    """모든 Python 파일에서 null 바이트를 제거하는 함수"""
    fixed_files = []
    
    for root, dirnames, filenames in os.walk(directory):
        for pattern in patterns:
            for filename in fnmatch.filter(filenames, pattern):
                file_path = os.path.join(root, filename)
                
                try:
                    # 파일 내용 읽기
                    with open(file_path, 'rb') as file:
                        content = file.read()
                    
                    # null 바이트 검사
                    if b'\x00' in content:
                        print(f"null 바이트 발견: {file_path}")
                        
                        # null 바이트 제거
                        content = content.replace(b'\x00', b'')
                        
                        # 새 파일로 저장
                        backup_path = file_path + '.bak'
                        os.rename(file_path, backup_path)
                        
                        with open(file_path, 'wb') as file:
                            file.write(content)
                        
                        fixed_files.append(file_path)
                        print(f"파일 수정 완료: {file_path}")
                        
                except Exception as e:
                    print(f"오류 발생: {file_path} - {str(e)}")
    
    return fixed_files

if __name__ == "__main__":
    # 현재 디렉토리를 기준으로 모든 Python 파일 검사
    fixed = remove_null_bytes('.')
    
    if fixed:
        print(f"\n총 {len(fixed)}개 파일 수정 완료:")
        for file in fixed:
            print(f"- {file}")
    else:
        print("\n수정할 파일이 없습니다.") 