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
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from utils.logger import logger
from config.settings import VECTOR_DB_PATH, OPENAI_API_KEY
from utils.pdf_processor import extract_text_from_pdf
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
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    
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
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    
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
    Search a vector database for documents similar to a query.
    
    Args:
        db_name: Name of the database to search
        query: Search query
        k: Number of results to return
        filter_metadata: Optional filter to apply to the metadata
        
    Returns:
        List[Document]: List of retrieved documents
    """
    logger.info(f"Searching vector database '{db_name}' with query: {query}")
    
    # Initialize embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    
    # Load vector store
    db_path = os.path.join(VECTOR_DB_PATH, db_name)
    if not os.path.exists(db_path):
        logger.error(f"Vector database '{db_name}' does not exist")
        return []
    
    vector_store = FAISS.load_local(db_path, embeddings)
    
    # Search
    if filter_metadata:
        results = vector_store.similarity_search(query, k=k, filter=filter_metadata)
    else:
        results = vector_store.similarity_search(query, k=k)
    
    logger.info(f"Found {len(results)} results for query: {query}")
    return results


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
    
    dbs = [d.name for d in db_dir.iterdir() if d.is_dir()]
    logger.info(f"Found {len(dbs)} vector databases")
    return dbs


def ensure_vector_db_directory():
    """벡터 DB 저장 디렉토리 확인 및 생성"""
    os.makedirs(VECTOR_DB_PATH, exist_ok=True)


def process_and_vectorize_paper(paper_info: Dict[str, Any], pdf_url: Optional[str] = None, 
                                collection_name: str = "research_papers"):
    """
    논문 정보와 PDF를 처리하여 벡터 DB에 저장
    
    Args:
        paper_info: 논문 메타데이터 (제목, 저자, 초록 등)
        pdf_url: PDF 다운로드 URL (없으면 로컬 파일만 시도)
        collection_name: 벡터 DB 컬렉션 이름
    
    Returns:
        처리 성공 여부
    """
    paper_id = paper_info.get("id") or paper_info.get("title", "")[:50].replace(" ", "_")
    logger.info(f"논문 처리 및 벡터화: {paper_id}")
    
    try:
        # 1. PDF 파일 확보 (로컬 또는 다운로드)
        pdf_path = get_local_pdf_path(paper_id, pdf_url or paper_info.get("pdf_url"))
        
        if not pdf_path:
            # PDF 없이 메타데이터만으로 처리
            logger.warning(f"PDF를 찾을 수 없음. 메타데이터만 벡터화: {paper_id}")
            metadata = {
                "title": paper_info.get("title", ""),
                "authors": paper_info.get("authors", []),
                "year": paper_info.get("year", ""),
                "abstract": paper_info.get("abstract", ""),
                "source": paper_info.get("source", "unknown"),
                "url": paper_info.get("url", ""),
                "paper_id": paper_id
            }
            
            # 메타데이터만으로 문서 생성
            content = f"제목: {metadata['title']}\n\n저자: {metadata['authors']}\n\n연도: {metadata['year']}\n\n초록: {metadata['abstract']}"
            documents = [Document(page_content=content, metadata=metadata)]
            
        else:
            # 2. PDF에서 텍스트 추출
            text = extract_text_from_pdf(pdf_path)
            
            if not text:
                logger.warning(f"PDF에서 텍스트를 추출할 수 없음: {pdf_path}")
                return False
            
            # 3. 문서 분할
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            
            splits = text_splitter.split_text(text)
            
            # 4. 메타데이터 준비
            metadata = {
                "title": paper_info.get("title", ""),
                "authors": paper_info.get("authors", []),
                "year": paper_info.get("year", ""),
                "abstract": paper_info.get("abstract", ""),
                "source": paper_info.get("source", "unknown"),
                "url": paper_info.get("url", ""),
                "pdf_path": pdf_path,
                "paper_id": paper_id
            }
            
            # 5. 문서 객체 생성
            documents = [
                Document(page_content=split, metadata=metadata) 
                for split in splits
            ]
        
        # 6. 벡터 DB에 저장
        db_path = os.path.join(VECTOR_DB_PATH, collection_name)
        
        # 기존 DB가 있는지 확인
        if os.path.exists(db_path):
            # 기존 DB에 추가
            embeddings = OpenAIEmbeddings()
            vectordb = FAISS.load_local(db_path, embeddings)
            vectordb.add_documents(documents)
            vectordb.save_local(db_path)
            logger.info(f"기존 벡터 DB에 문서 추가됨: {collection_name}, 문서 ID: {paper_id}")
        else:
            # 새 DB 생성
            create_vector_db(documents, collection_name)
        
        # 7. 처리된 논문 ID 저장 (중복 처리 방지)
        processed_papers_path = os.path.join(VECTOR_DB_PATH, "processed_papers.json")
        processed_papers = set()
        
        if os.path.exists(processed_papers_path):
            with open(processed_papers_path, 'r', encoding='utf-8') as f:
                processed_papers = set(json.load(f))
        
        processed_papers.add(paper_id)
        
        with open(processed_papers_path, 'w', encoding='utf-8') as f:
            json.dump(list(processed_papers), f, ensure_ascii=False)
        
        logger.info(f"논문 처리 및 벡터화 완료: {paper_id}")
        return True
    
    except Exception as e:
        logger.error(f"논문 처리 및 벡터화 실패: {str(e)}", exc_info=True)
        return False