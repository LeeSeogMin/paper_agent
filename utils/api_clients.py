"""
API client utilities for academic research APIs.

This module provides functions to interact with various academic research APIs
such as Semantic Scholar and Google Scholar.
"""

import os
import json
import time
import requests
from typing import List, Dict, Any, Optional

from utils.logger import logger
from config.api_keys import SEMANTIC_SCHOLAR_API_KEY, GOOGLE_SCHOLAR_API_KEY


def search_academic_papers(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Search for papers on Semantic Scholar and Google Scholar.
    
    Args:
        query (str): Search query
        max_results (int, optional): Maximum number of results to return. Default is 5.
        
    Returns:
        List[Dict[str, Any]]: List of paper metadata
        
    Raises:
        Exception: If all search APIs fail
    """
    semantic_scholar_error = None
    google_scholar_error = None
    
    # 1. Try Semantic Scholar search
    try:
        logger.info(f"Searching Semantic Scholar for: {query}")
        results = semantic_scholar_search(query, max_results)
        if results:
            logger.info(f"Found {len(results)} papers on Semantic Scholar")
            return results
    except Exception as e:
        semantic_scholar_error = str(e)
        logger.warning(f"Semantic Scholar search failed: {semantic_scholar_error}")
    
    # 2. Try Google Scholar search
    try:
        logger.info(f"Searching Google Scholar for: {query}")
        results = google_scholar_search(query, max_results)
        if results:
            logger.info(f"Found {len(results)} papers on Google Scholar")
            return results
    except Exception as e:
        google_scholar_error = str(e)
        logger.warning(f"Google Scholar search failed: {google_scholar_error}")
    
    # 3. If both searches fail, raise an error
    error_message = f"All academic search APIs failed. Semantic Scholar: {semantic_scholar_error}, Google Scholar: {google_scholar_error}"
    logger.error(error_message)
    raise Exception(error_message)


def semantic_scholar_search(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Search for papers on Semantic Scholar.
    
    Args:
        query (str): Search query
        max_results (int, optional): Maximum number of results to return. Default is 5.
        
    Returns:
        List[Dict[str, Any]]: List of paper metadata
    """
    logger.info(f"Searching Semantic Scholar for: {query}")
    
    # API endpoint
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    
    # Request parameters
    params = {
        "query": query,
        "limit": max_results,
        "fields": "title,authors,year,abstract,url,venue,citationCount,paperAbstract,externalIds,openAccessPdf"
    }
    
    # Set up headers
    headers = {}
    if SEMANTIC_SCHOLAR_API_KEY:
        headers["x-api-key"] = SEMANTIC_SCHOLAR_API_KEY
    
    try:
        # Send request
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        
        # Parse response
        data = response.json()
        papers = data.get("data", [])
        
        # Format results
        results = []
        for paper in papers:
            # Extract paper metadata
            paper_id = paper.get("paperId", "")
            title = paper.get("title", "")
            abstract = paper.get("abstract", "") or paper.get("paperAbstract", "")
            
            # Extract authors
            authors = []
            for author in paper.get("authors", []):
                name = author.get("name", "")
                if name:
                    authors.append(name)
            
            # Extract year
            year = paper.get("year")
            
            # Extract venue
            venue = paper.get("venue", "")
            
            # Extract citation count
            citation_count = paper.get("citationCount", 0)
            
            # Extract URL
            url = paper.get("url", "")
            
            # Extract PDF URL if available
            pdf_url = ""
            if "openAccessPdf" in paper and paper["openAccessPdf"]:
                pdf_url = paper["openAccessPdf"].get("url", "")
            
            # Add to results
            results.append({
                "paperId": paper_id,
                "title": title,
                "abstract": abstract,
                "authors": authors,
                "year": year,
                "venue": venue,
                "citationCount": citation_count,
                "url": url,
                "pdfUrl": pdf_url,
                "source": "Semantic Scholar"
            })
        
        logger.info(f"Found {len(results)} papers on Semantic Scholar")
        return results
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Error searching Semantic Scholar: {str(e)}")
        raise Exception(f"Semantic Scholar search failed: {str(e)}")


def google_scholar_search(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Search for papers on Google Scholar.
    
    Note: This function is a placeholder. Google Scholar does not offer an official API,
    and web scraping may violate their terms of service. Consider using Semantic Scholar
    or other academic APIs instead.
    
    Args:
        query (str): Search query
        max_results (int, optional): Maximum number of results to return. Default is 5.
        
    Returns:
        List[Dict[str, Any]]: List of paper metadata
    """
    logger.warning("Google Scholar search is a placeholder and does not make actual API calls.")
    
    # Return empty list as this is just a placeholder
    return []


def download_pdf(url: str, output_path: str) -> bool:
    """
    Download a PDF from a URL.
    
    Args:
        url (str): URL to the PDF
        output_path (str): Path to save the PDF to
        
    Returns:
        bool: True if successful, False otherwise
    """
    logger.info(f"Downloading PDF from {url}")
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        logger.info(f"PDF downloaded to {output_path}")
        return True
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading PDF: {str(e)}")
        return False
    except IOError as e:
        logger.error(f"Error writing PDF to file: {str(e)}")
        return False