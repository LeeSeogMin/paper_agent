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