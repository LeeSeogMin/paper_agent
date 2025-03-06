"""
자료정리 에이전트 모듈
수집된 PDF 논문을 파싱하고 벡터 DB에 저장하는 에이전트입니다.
"""

import os
import re
import json
from typing import Dict, Any, List, Optional, Tuple, Union
from pydantic import BaseModel, Field

from langchain_core.runnables import RunnableConfig
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from config.settings import VECTOR_DB_DIR, PDF_STORAGE_DIR, CHUNK_SIZE, CHUNK_OVERLAP
from config.api_keys import OPENAI_API_KEY, XAI_API_KEY
from utils.logger import logger
from utils.pdf_processor import PDFProcessor
from models.research import ResearchMaterial
from agents.base import BaseAgent


class ProcessingResult(BaseModel):
    """PDF 처리 결과 형식"""
    pdf_path: str = Field(description="처리된 PDF 파일 경로")
    total_pages: int = Field(description="총 페이지 수")
    total_chunks: int = Field(description="생성된 텍스트 청크 수")
    extracted_text_length: int = Field(description="추출된 텍스트 길이")
    metadata: Dict[str, Any] = Field(description="PDF 메타데이터")


class VectorDBInfo(BaseModel):
    """벡터 DB 정보 형식"""
    collection_name: str = Field(description="컬렉션 이름")
    document_count: int = Field(description="저장된 문서 수")
    embedding_dimension: int = Field(description="임베딩 차원")
    metadata_fields: List[str] = Field(description="메타데이터 필드 목록")


class ProcessingStats(BaseModel):
    """처리 통계 형식"""
    total_pdfs: int = Field(description="처리된 PDF 파일 수")
    total_pages: int = Field(description="처리된 총 페이지 수")
    total_chunks: int = Field(description="생성된 총 청크 수")
    total_tokens: int = Field(description="처리된 총 토큰 수")
    average_chunks_per_pdf: float = Field(description="PDF당 평균 청크 수")
    processing_time: float = Field(description="처리 시간 (초)")


class DataProcessingAgent(BaseAgent[List[Document]]):
    """PDF 파싱 및 벡터 DB 저장 에이전트"""

    def __init__(
        self,
        name: str = "자료정리 에이전트",
        description: str = "PDF 파싱 및 벡터 DB 저장",
        vector_db_dir: str = VECTOR_DB_DIR,
        pdf_storage_dir: str = PDF_STORAGE_DIR,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        embedding_model: str = "all-mpnet-base-v2",
        verbose: bool = False
    ):
        """
        DataProcessingAgent 초기화

        Args:
            name (str, optional): 에이전트 이름. 기본값은 "자료정리 에이전트"
            description (str, optional): 에이전트 설명. 기본값은 "PDF 파싱 및 벡터 DB 저장"
            vector_db_dir (str, optional): 벡터 DB 저장 디렉토리. 기본값은 VECTOR_DB_DIR
            pdf_storage_dir (str, optional): PDF 저장 디렉토리. 기본값은 PDF_STORAGE_DIR
            chunk_size (int, optional): 청크 크기. 기본값은 CHUNK_SIZE
            chunk_overlap (int, optional): 청크 오버랩. 기본값은 CHUNK_OVERLAP
            embedding_model (str, optional): 임베딩 모델. 기본값은 "all-mpnet-base-v2"
            verbose (bool, optional): 상세 로깅 활성화 여부. 기본값은 False
        """
        super().__init__(name, description, verbose=verbose)
        
        # 디렉토리 생성
        os.makedirs(vector_db_dir, exist_ok=True)
        os.makedirs(pdf_storage_dir, exist_ok=True)
        
        # 설정 저장
        self.vector_db_dir = vector_db_dir
        self.pdf_storage_dir = pdf_storage_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = embedding_model
        
        # PDF 처리기 초기화
        self.pdf_processor = PDFProcessor(verbose=verbose)
        
        # 텍스트 분할기 초기화 (계층적 분할 전략)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=[
                "\n\n## ",  # 주요 섹션 헤딩 (Markdown-style)
                "\n\n",     # 문단 구분
                "\n",       # 개행 문자
                ". ",       # 문장 종료
                "! ",       # 감탄문 종료
                "? ",       # 질문 종료
                "; ",       # 세미콜론 구분
                ": ",       # 콜론 구분
                ", ",       # 쉼표 구분
                " ",        # 단어 구분
                ""          # 최종 문자 단위
            ],
            keep_separator=True  # 분할 기호 유지
        )
        
        # 임베딩 객체 초기화
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model
        )
        
        # 벡터 DB 초기화 (사용 시 생성)
        self.vectordb = None
        self.collection_name = None
        
        # 처리 통계 초기화
        self.stats = {
            "total_pdfs": 0,
            "total_pages": 0,
            "total_chunks": 0,
            "total_tokens": 0,
            "processing_times": []
        }
        
        logger.info(f"{self.name} 초기화 완료: 청크 크기 {self.chunk_size}, 오버랩 {self.chunk_overlap}")

    def process_pdf(self, pdf_path: str) -> ProcessingResult:
        """
        PDF 파일을 처리합니다.

        Args:
            pdf_path (str): PDF 파일 경로

        Returns:
            ProcessingResult: PDF 처리 결과
        """
        import time
        start_time = time.time()
        
        logger.info(f"PDF 처리 중: {pdf_path}")
        
        try:
            # PDF 처리
            _, text, pages, metadata = self.pdf_processor.process_file(pdf_path)
            
            if not text:
                raise ValueError(f"PDF에서 텍스트를 추출할 수 없습니다: {pdf_path}")
            
            # 통계 업데이트
            self.stats["total_pdfs"] += 1
            self.stats["total_pages"] += len(pages) if pages else 1
            
            # 처리 시간 기록
            processing_time = time.time() - start_time
            self.stats["processing_times"].append(processing_time)
            
            logger.info(f"PDF 처리 완료: {pdf_path}, {len(pages)}페이지, {len(text)}자")
            
            # 청크 생성
            chunks = self.create_chunks(text, metadata=metadata)
            
            return ProcessingResult(
                pdf_path=pdf_path,
                total_pages=len(pages) if pages else 1,
                total_chunks=len(chunks),
                extracted_text_length=len(text),
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"PDF 처리 중 오류 발생: {str(e)}")
            raise

    def create_chunks(self, text: str, metadata: Dict[str, Any] = None) -> List[Document]:
        """
        텍스트를 청크로 분할합니다.

        Args:
            text (str): 분할할 텍스트
            metadata (Dict[str, Any], optional): 추가할 메타데이터

        Returns:
            List[Document]: 생성된 문서 청크 목록
        """
        logger.info(f"텍스트 청크 생성 중: {len(text)}자")
        
        try:
            # 메타데이터 준비
            metadata = metadata or {}
            
            # 텍스트 분할
            chunks = self.text_splitter.create_documents(
                texts=[text],
                metadatas=[metadata]
            )
            
            # 통계 업데이트
            self.stats["total_chunks"] += len(chunks)
            
            # 토큰 수 추정 (영어 기준 대략적 추정)
            estimated_tokens = sum(len(chunk.page_content.split()) * 1.3 for chunk in chunks)
            self.stats["total_tokens"] += int(estimated_tokens)
            
            logger.info(f"청크 생성 완료: {len(chunks)}개 청크")
            return chunks
            
        except Exception as e:
            logger.error(f"청크 생성 중 오류 발생: {str(e)}")
            raise

    def initialize_vector_db(self, collection_name: str) -> Chroma:
        """
        벡터 DB를 초기화합니다.

        Args:
            collection_name (str): 컬렉션 이름

        Returns:
            Chroma: 초기화된 벡터 DB 객체
        """
        logger.info(f"벡터 DB 초기화 중: 컬렉션 '{collection_name}'")
        
        try:
            # 벡터 DB 초기화
            vectordb = Chroma(
                collection_name=collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.vector_db_dir
            )
            
            self.vectordb = vectordb
            self.collection_name = collection_name
            
            logger.info(f"벡터 DB 초기화 완료: 컬렉션 '{collection_name}'")
            return vectordb
            
        except Exception as e:
            logger.error(f"벡터 DB 초기화 중 오류 발생: {str(e)}")
            raise
            
    def add_documents(self, documents: List[Document]) -> int:
        """
        벡터 DB에 문서를 추가합니다.

        Args:
            documents (List[Document]): 추가할 문서 목록

        Returns:
            int: 추가된 문서 수
        """
        if not self.vectordb:
            raise ValueError("벡터 DB가 초기화되지 않았습니다. initialize_vector_db()를 먼저 호출하세요.")
            
        logger.info(f"{len(documents)}개 문서를 벡터 DB에 추가 중...")
        
        try:
            # 문서 추가
            self.vectordb.add_documents(documents)
            
            # 변경사항 저장
            if hasattr(self.vectordb, "persist"):
                self.vectordb.persist()
                
            logger.info(f"{len(documents)}개 문서가 벡터 DB에 추가됨")
            return len(documents)
            
        except Exception as e:
            logger.error(f"문서 추가 중 오류 발생: {str(e)}")
            raise
            
    def process_research_material(self, material: ResearchMaterial) -> List[Document]:
        """
        연구 자료를 처리하고 벡터 DB에 저장합니다.

        Args:
            material (ResearchMaterial): 처리할 연구 자료

        Returns:
            List[Document]: 생성된 문서 목록
        """
        logger.info(f"연구 자료 처리 중: {material.title}")
        
        documents = []
        
        try:
            # PDF 파일인 경우
            if material.local_path and os.path.exists(material.local_path) and material.local_path.lower().endswith('.pdf'):
                # PDF 처리
                result = self.process_pdf(material.local_path)
                
                # 메타데이터 준비
                metadata = {
                    "title": material.title,
                    "authors": ', '.join(material.authors) if material.authors else "",
                    "year": material.year or "",
                    "source": material.source or "",
                    "url": material.url or "",
                    "local_path": material.local_path,
                    "is_pdf": True
                }
                
                # PDF에서 텍스트 추출
                _, text, _, _ = self.pdf_processor.process_file(material.local_path)
                
                # 청크 생성
                if text:
                    documents = self.create_chunks(text, metadata=metadata)
                    
            # 텍스트 콘텐츠가 있는 경우 (PDF가 아니거나 추가적인 콘텐츠)
            elif material.content:
                # 메타데이터 준비
                metadata = {
                    "title": material.title,
                    "authors": ', '.join(material.authors) if material.authors else "",
                    "year": material.year or "",
                    "source": material.source or "",
                    "url": material.url or "",
                    "is_pdf": False
                }
                
                # 청크 생성
                documents = self.create_chunks(material.content, metadata=metadata)
                
            # 벡터 DB에 추가 (초기화된 경우)
            if self.vectordb and documents:
                self.add_documents(documents)
                
            logger.info(f"연구 자료 처리 완료: {material.title}, {len(documents)}개 청크 생성")
            return documents
            
        except Exception as e:
            logger.error(f"연구 자료 처리 중 오류 발생: {str(e)}")
            return []
            
    def process_materials(self, materials: List[ResearchMaterial], collection_name: str) -> List[Document]:
        """
        여러 연구 자료를 처리하고 벡터 DB에 저장합니다.

        Args:
            materials (List[ResearchMaterial]): 처리할 연구 자료 목록
            collection_name (str): 벡터 DB 컬렉션 이름

        Returns:
            List[Document]: 생성된 문서 목록
        """
        import time
        start_time = time.time()
        
        logger.info(f"{len(materials)}개 연구 자료 처리 시작")
        
        # 벡터 DB 초기화
        self.initialize_vector_db(collection_name)
        
        all_documents = []
        
        # 각 자료 처리
        for material in materials:
            try:
                documents = self.process_research_material(material)
                all_documents.extend(documents)
            except Exception as e:
                logger.error(f"자료 '{material.title}' 처리 중 오류 발생: {str(e)}")
                continue
        
        # 처리 시간 계산
        processing_time = time.time() - start_time
        
        # 통계 업데이트
        avg_chunks = self.stats["total_chunks"] / self.stats["total_pdfs"] if self.stats["total_pdfs"] > 0 else 0
        
        stats = ProcessingStats(
            total_pdfs=self.stats["total_pdfs"],
            total_pages=self.stats["total_pages"],
            total_chunks=self.stats["total_chunks"],
            total_tokens=self.stats["total_tokens"],
            average_chunks_per_pdf=avg_chunks,
            processing_time=processing_time
        )
        
        logger.info(f"자료 처리 완료: {len(all_documents)}개 청크 생성, 처리 시간: {processing_time:.2f}초")
        
        # 상태 업데이트
        self.update_state({
            "stats": stats.dict(),
            "collection_name": collection_name,
            "document_count": len(all_documents)
        })
        
        return all_documents
        
    def search_documents(self, query: str, top_k: int = 5) -> List[Document]:
        """
        벡터 DB에서 문서를 검색합니다.

        Args:
            query (str): 검색 쿼리
            top_k (int, optional): 반환할 최대 문서 수. 기본값은 5

        Returns:
            List[Document]: 검색된 문서 목록
        """
        if not self.vectordb:
            raise ValueError("벡터 DB가 초기화되지 않았습니다. initialize_vector_db()를 먼저 호출하세요.")
            
        logger.info(f"쿼리 '{query}'로 문서 검색 중...")
        
        try:
            # 유사도 검색 수행
            documents = self.vectordb.similarity_search(query, k=top_k)
            
            logger.info(f"검색 완료: {len(documents)}개 문서 발견")
            return documents
            
        except Exception as e:
            logger.error(f"문서 검색 중 오류 발생: {str(e)}")
            return []
            
    def get_vector_db_info(self) -> VectorDBInfo:
        """
        현재 벡터 DB 정보를 가져옵니다.

        Returns:
            VectorDBInfo: 벡터 DB 정보
        """
        if not self.vectordb:
            raise ValueError("벡터 DB가 초기화되지 않았습니다.")
            
        try:
            # 벡터 DB 정보 수집
            collection_name = self.collection_name or "unknown"
            
            # 문서 수 추정 (Chroma API에 따라 다를 수 있음)
            document_count = 0
            if hasattr(self.vectordb, "_collection") and hasattr(self.vectordb._collection, "count"):
                document_count = self.vectordb._collection.count()
            
            # 임베딩 차원 (기본 OpenAI ada-002 모델은 1536차원)
            embedding_dimension = 768  # all-mpnet-base-v2 모델은 768차원
            
            # 메타데이터 필드
            metadata_fields = ["title", "authors", "year", "source", "url", "is_pdf"]
            
            return VectorDBInfo(
                collection_name=collection_name,
                document_count=document_count,
                embedding_dimension=embedding_dimension,
                metadata_fields=metadata_fields
            )
            
        except Exception as e:
            logger.error(f"벡터 DB 정보 가져오기 중 오류 발생: {str(e)}")
            
            # 기본 정보 반환
            return VectorDBInfo(
                collection_name=self.collection_name or "unknown",
                document_count=0,
                embedding_dimension=768,
                metadata_fields=[]
            )
            
    def run(
        self, 
        materials: List[ResearchMaterial], 
        collection_name: Optional[str] = None,
        config: Optional[RunnableConfig] = None
    ) -> List[Document]:
        """
        자료정리 에이전트를 실행하고 문서를 반환합니다.

        Args:
            materials (List[ResearchMaterial]): 처리할 연구 자료 목록
            collection_name (Optional[str], optional): 벡터 DB 컬렉션 이름
            config (Optional[RunnableConfig], optional): 실행 구성

        Returns:
            List[Document]: 생성된 문서 목록
        """
        logger.info(f"자료정리 에이전트 실행 중: {len(materials)}개 자료")
        
        # 컬렉션 이름 생성 (제공되지 않은 경우)
        if not collection_name:
            import uuid
            collection_name = f"research_{uuid.uuid4().hex[:8]}"
        
        try:
            # 자료 처리 및 벡터 DB 저장
            documents = self.process_materials(materials, collection_name)
            
            logger.info(f"자료정리 에이전트 실행 완료: {len(documents)}개 청크 생성됨")
            return documents
            
        except Exception as e:
            logger.error(f"자료정리 에이전트 실행 중 오류 발생: {str(e)}")
            return []
