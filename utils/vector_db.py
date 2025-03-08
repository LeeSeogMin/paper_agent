"""
Vector database utilities for research content storage and retrieval.

This module provides functions to create, update, and search a vector database
for efficient retrieval of research content.
"""

import os
import json
from typing import List, Dict, Any, Optional, Union
from pathlib import Path

import numpy as np
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

from utils.logger import logger
from config.settings import VECTOR_DB_PATH, OPENAI_API_KEY
from utils.pdf_downloader import get_local_pdf_path


def create_vector_db(documents: List[Union[Document, Dict[str, Any]]], db_name: str) -> str:
    """
    Create a vector database from a list of documents.
    
    Args:
        documents: List of documents or dictionaries to add to the vector database
        db_name: Name of the database to create
        
    Returns:
        str: Path to the created vector database
    """
    logger.info(f"Creating vector database '{db_name}' with {len(documents)} documents")
    
    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
    
    # Convert dictionaries to Document objects if needed
    docs = []
    for doc in documents:
        if isinstance(doc, dict):
            content = doc.get("content", "")
            metadata = {k: v for k, v in doc.items() if k != "content"}
            docs.append(Document(page_content=content, metadata=metadata))
        else:
            docs.append(doc)
    
    # Create vector store
    vector_store = FAISS.from_documents(docs, embeddings)
    
    # Create database directory if it doesn't exist
    db_dir = Path(VECTOR_DB_PATH)
    db_dir.mkdir(parents=True, exist_ok=True)
    
    # Save vector store to disk
    db_path = os.path.join(VECTOR_DB_PATH, db_name)
    vector_store.save_local(db_path)
    
    logger.info(f"Vector database created at {db_path}")
    return db_path


def update_vector_db(db_name: str, new_documents: List[Union[Document, Dict[str, Any]]]) -> str:
    """
    Update an existing vector database with new documents.
    
    Args:
        db_name: Name of the database to update
        new_documents: List of new documents to add
        
    Returns:
        str: Path to the updated vector database
    """
    logger.info(f"Updating vector database '{db_name}' with {len(new_documents)} new documents")
    
    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
    
    # Load existing vector store
    db_path = os.path.join(VECTOR_DB_PATH, db_name)
    if os.path.exists(db_path):
        vector_store = FAISS.load_local(db_path, embeddings)
    else:
        logger.warning(f"Vector database '{db_name}' does not exist, creating new one")
        return create_vector_db(new_documents, db_name)
    
    # Convert dictionaries to Document objects if needed
    docs = []
    for doc in new_documents:
        if isinstance(doc, dict):
            content = doc.get("content", "")
            metadata = {k: v for k, v in doc.items() if k != "content"}
            docs.append(Document(page_content=content, metadata=metadata))
        else:
            docs.append(doc)
    
    # Add new documents
    vector_store.add_documents(docs)
    
    # Save updated vector store
    vector_store.save_local(db_path)
    
    logger.info(f"Vector database updated at {db_path}")
    return db_path


def search_vector_db(
    db_name: str, 
    query: str, 
    k: int = 5,
    filter_metadata: Optional[Dict[str, Any]] = None
) -> List[Document]:
    """
    벡터 데이터베이스에서 문서를 검색합니다.
    
    Args:
        db_name: 검색할 데이터베이스 이름
        query: 검색 쿼리
        k: 반환할 최대 결과 수
        filter_metadata: 메타데이터 필터링 조건
        
    Returns:
        List[Document]: 검색된 문서 목록
    """
    logger.info(f"벡터 DB '{db_name}' 검색 중: {query}")
    
    try:
        # 벡터 DB 디렉토리 확인
        db_path = os.path.join(VECTOR_DB_PATH, db_name)
        if not os.path.exists(db_path):
            logger.error(f"벡터 DB가 존재하지 않음: {db_path}")
            return []
            
        # 임베딩 초기화
        embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
        
        # Chroma DB 로드
        vectorstore = Chroma(
            persist_directory=db_path,
            embedding_function=embeddings,
            collection_name=db_name
        )
        
        # 검색 수행
        if filter_metadata:
            results = vectorstore.similarity_search(
                query,
                k=k,
                filter=filter_metadata
            )
        else:
            results = vectorstore.similarity_search(
                query,
                k=k
            )
        
        logger.info(f"검색 완료: {len(results)}개 결과")
        return results
        
    except Exception as e:
        logger.error(f"벡터 DB 검색 중 오류 발생: {str(e)}")
        return []


def delete_vector_db(db_name: str) -> bool:
    """
    Delete a vector database.
    
    Args:
        db_name: Name of the database to delete
        
    Returns:
        bool: True if successful, False otherwise
    """
    logger.info(f"Deleting vector database '{db_name}'")
    
    db_path = os.path.join(VECTOR_DB_PATH, db_name)
    if not os.path.exists(db_path):
        logger.warning(f"Vector database '{db_name}' does not exist")
        return False
    
    try:
        import shutil
        shutil.rmtree(db_path)
        logger.info(f"Vector database '{db_name}' deleted")
        return True
    except Exception as e:
        logger.error(f"Error deleting vector database '{db_name}': {str(e)}")
        return False


def list_vector_dbs() -> List[str]:
    """
    List all available vector databases.
    
    Returns:
        List[str]: List of database names
    """
    logger.info("Listing vector databases")
    
    db_dir = Path(VECTOR_DB_PATH)
    if not db_dir.exists():
        logger.warning("Vector database directory does not exist")
        return []
    
    # 디렉토리 목록 확인
    dbs = [d.name for d in db_dir.iterdir() if d.is_dir()]
    
    # research_papers 디렉토리가 있는지 확인
    research_papers_dir = db_dir / "research_papers"
    if research_papers_dir.exists() and research_papers_dir.is_dir():
        if "research_papers" not in dbs:
            dbs.append("research_papers")
    
    logger.info(f"Found {len(dbs)} vector databases")
    return dbs


def ensure_vector_db_directory():
    """벡터 DB 저장 디렉토리 확인 및 생성"""
    os.makedirs(VECTOR_DB_PATH, exist_ok=True)


def process_and_vectorize_paper(pdf_path: str) -> Dict[str, Any]:
    """
    PDF 파일을 처리하고 벡터화하여 저장
    
    Args:
        pdf_path: PDF 파일 경로
        
    Returns:
        Dict[str, Any]: 처리된 논문 정보
    """
    from utils.pdf_processor import extract_text_from_pdf
    import os
    import uuid
    import re
    import hashlib
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import Chroma
    
    try:
        # 논문 ID 설정
        paper_id = f"paper_{uuid.uuid4().hex[:8]}"
        
        # PDF 텍스트 추출
        text = extract_text_from_pdf(pdf_path)
        if not text or len(text) < 100:
            logger.warning(f"PDF 텍스트 추출 실패 또는 내용 부족: {pdf_path}")
            return None
        
        # 중복 체크 - 해시 값 생성
        content_hash = hashlib.md5(text[:5000].encode()).hexdigest()
        
        # 이미 저장된 해시 값 확인
        hash_file = os.path.join(VECTOR_DB_PATH, "content_hashes.json")
        existing_hashes = {}
        if os.path.exists(hash_file):
            with open(hash_file, 'r') as f:
                existing_hashes = json.load(f)
        
        # 중복 확인
        if content_hash in existing_hashes:
            logger.info(f"중복 문서 발견, 기존 ID 사용: {existing_hashes[content_hash]}")
            return {
                "id": existing_hashes[content_hash],
                "title": "...",  # 기존 문서 정보 반환
                "is_duplicate": True
            }
        
        # 메타데이터 추출 (기본 정보)
        file_name = os.path.basename(pdf_path)
        title = os.path.splitext(file_name)[0].replace('_', ' ')
        
        # 정규식으로 제목과 저자 추출 시도
        title_match = re.search(r'(?:Title|TITLE):\s*([^\n]+)', text[:2000])
        if title_match:
            title = title_match.group(1).strip()
        
        authors = ""
        authors_match = re.search(r'(?:Author|AUTHORS|authors)s?:\s*([^\n]+)', text[:2000])
        if authors_match:
            authors = authors_match.group(1).strip()
        
        # 초록 추출 시도
        abstract = ""
        abstract_match = re.search(r'(?:Abstract|ABSTRACT):\s*([^\n]+(?:\n[^\n]+){1,10})', text[:5000])
        if abstract_match:
            abstract = abstract_match.group(1).strip()
        
        # 텍스트 분할
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200
        )
        chunks = text_splitter.split_text(text)
        
        # 벡터 DB 저장 경로 생성
        os.makedirs(VECTOR_DB_PATH, exist_ok=True)
        collection_name = "research_papers"  # 모든 논문을 하나의 컬렉션에 저장
        
        # research_papers 디렉토리 명시적으로 생성
        research_papers_dir = os.path.join(VECTOR_DB_PATH, "research_papers")
        os.makedirs(research_papers_dir, exist_ok=True)
        
        # 임베딩 및 벡터 DB 저장
        embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
        metadata = [{"source": pdf_path, "paper_id": paper_id, "chunk": i, "title": title} for i in range(len(chunks))]
        
        # 기존 컬렉션이 있는지 확인
        try:
            # 컬렉션 이름을 명시적으로 지정
            persist_directory = research_papers_dir
            
            vectorstore = Chroma(
                persist_directory=persist_directory,
                embedding_function=embeddings,
                collection_name=collection_name
            )
            # 기존 컬렉션에 새 문서 추가
            vectorstore.add_texts(
                texts=chunks,
                metadatas=metadata
            )
        except Exception as e:
            # 컬렉션이 없으면 새로 생성
            logger.info(f"Creating new vector collection: {collection_name}")
            vectorstore = Chroma.from_texts(
                texts=chunks,
                embedding=embeddings,
                metadatas=metadata,
                persist_directory=persist_directory,
                collection_name=collection_name
            )
        
        vectorstore.persist()
        
        # 논문 메타데이터 반환
        paper_info = {
            "id": paper_id,
            "title": title,
            "authors": authors,
            "abstract": abstract,
            "pdf_path": pdf_path,
            "vector_collection": collection_name,
            "content_length": len(text),
            "chunk_count": len(chunks)
        }
        
        # 해시 값 저장
        existing_hashes[content_hash] = paper_id
        with open(hash_file, 'w') as f:
            json.dump(existing_hashes, f)
        
        return paper_info
        
    except Exception as e:
        logger.error(f"PDF 벡터화 중 오류: {str(e)}", exc_info=True)
        return None


def vectorize_content(content: str, title: str, material_id: str) -> Dict[str, Any]:
    """
    텍스트 콘텐츠를 벡터화하여 저장
    
    Args:
        content: 벡터화할 텍스트 콘텐츠
        title: 문서 제목
        material_id: 문서 ID
        
    Returns:
        Dict[str, Any]: 처리된 문서 정보
    """
    import os
    import hashlib
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import Chroma
    
    try:
        if not content or len(content) < 100:
            logger.warning(f"문서 내용이 부족합니다: {title}")
            return None
        
        # 중복 체크 - 해시 값 생성
        content_hash = hashlib.md5(content[:5000].encode()).hexdigest()
        
        # 이미 저장된 해시 값 확인
        hash_file = os.path.join(VECTOR_DB_PATH, "content_hashes.json")
        existing_hashes = {}
        if os.path.exists(hash_file):
            with open(hash_file, 'r') as f:
                existing_hashes = json.load(f)
        
        # 중복 확인
        if content_hash in existing_hashes:
            logger.info(f"중복 문서 발견, 기존 ID 사용: {existing_hashes[content_hash]}")
            return {
                "id": existing_hashes[content_hash],
                "title": title,
                "is_duplicate": True
            }
        
        # 텍스트 분할
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200
        )
        chunks = text_splitter.split_text(content)
        
        # 벡터 DB 저장 경로 생성
        os.makedirs(VECTOR_DB_PATH, exist_ok=True)
        collection_name = "research_papers"  # 모든 논문을 하나의 컬렉션에 저장
        
        # research_papers 디렉토리 명시적으로 생성
        research_papers_dir = os.path.join(VECTOR_DB_PATH, "research_papers")
        os.makedirs(research_papers_dir, exist_ok=True)
        
        # 임베딩 및 벡터 DB 저장
        embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
        metadata = [{"source": "text", "paper_id": material_id, "chunk": i, "title": title} for i in range(len(chunks))]
        
        # 기존 컬렉션이 있는지 확인
        try:
            # 컬렉션 이름을 명시적으로 지정
            persist_directory = research_papers_dir
            
            vectorstore = Chroma(
                persist_directory=persist_directory,
                embedding_function=embeddings,
                collection_name=collection_name
            )
            # 기존 컬렉션에 새 문서 추가
            vectorstore.add_texts(
                texts=chunks,
                metadatas=metadata
            )
        except Exception as e:
            # 컬렉션이 없으면 새로 생성
            logger.info(f"Creating new vector collection: {collection_name}")
            vectorstore = Chroma.from_texts(
                texts=chunks,
                embedding=embeddings,
                metadatas=metadata,
                persist_directory=persist_directory,
                collection_name=collection_name
            )
        
        vectorstore.persist()
        
        # 해시 저장
        existing_hashes[content_hash] = material_id
        with open(hash_file, 'w') as f:
            json.dump(existing_hashes, f)
        
        # 문서 메타데이터 반환
        return {
            "id": material_id,
            "title": title,
            "vector_collection": collection_name,
            "content_length": len(content),
            "chunks": len(chunks)
        }
        
    except Exception as e:
        logger.error(f"텍스트 벡터화 중 오류 발생: {str(e)}")
        return None