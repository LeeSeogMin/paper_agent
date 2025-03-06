import os
from typing import List, Dict, Any, Optional
from .academic_search import AcademicSearchManager

class RAGEnhancer:
    """
    RAG(Retrieval-Augmented Generation) system that integrates academic search functionality
    """
    
    def __init__(self):
        self.search_manager = AcademicSearchManager()
    
    def enhance_prompt_with_research(self, 
                                    topic: str, 
                                    base_prompt: str,
                                    num_sources: int = 5) -> str:
        """
        Enhance a prompt with academic search results for a given topic
        
        Args:
            topic: Topic to search for
            base_prompt: Base prompt to enhance
            num_sources: Number of sources to add
        
        Returns:
            Enhanced prompt
        """
        # Perform academic search
        search_results = self.search_manager.search(
            query=topic,
            source="all",
            limit=num_sources
        )
        
        # Extract citation information
        citations = self.search_manager.get_citations_for_rag(search_results)
        
        # Construct context to add to prompt
        context = "## Relevant Academic Sources:\n\n"
        
        for i, citation in enumerate(citations, 1):
            context += f"{i}. {citation['citation_text']}\n"
            if citation['snippet']:
                snippet = citation['snippet']
                if len(snippet) > 300:  # Truncate long snippets
                    snippet = snippet[:300] + "..."
                context += f"   Summary: {snippet}\n"
            context += "\n"
        
        # Add citation instructions
        citation_instructions = """
## Citation and Reference Requirements:
1. You MUST cite the sources above when using information from them.
2. All claims and statements must be supported by citations.
3. Use the format (Author, Year) for in-text citations.
4. Include a complete bibliography/references section at the end.
5. All content must be written in English.
"""
        
        # Construct enhanced prompt
        enhanced_prompt = f"{base_prompt}\n\n{context}\n{citation_instructions}\n"
        enhanced_prompt += "Please use the academic sources above to inform your response and cite them appropriately."
        
        return enhanced_prompt
    
    def get_research_summary(self, topic: str, limit: int = 10) -> str:
        """
        Generate a research summary for a topic
        
        Args:
            topic: Topic to search for
            limit: Number of search results
            
        Returns:
            Research summary string
        """
        # Perform academic search
        search_results = self.search_manager.search(
            query=topic,
            source="all",
            limit=limit
        )
        
        # Format search results
        formatted_results = self.search_manager.format_search_results(search_results)
        
        return formatted_results
    
    def get_citation_data(self, topic: str, limit: int = 15) -> List[Dict[str, Any]]:
        """
        Get citation data for a topic to be used in papers and reports
        
        Args:
            topic: Topic to search for
            limit: Number of search results
            
        Returns:
            List of citation data dictionaries
        """
        # Perform academic search
        search_results = self.search_manager.search(
            query=topic,
            source="all",
            limit=limit
        )
        
        # Extract citation information in a format suitable for references
        citations = []
        
        for source_type, results in search_results.items():
            if source_type == "query" or not results:
                continue
                
            for result in results:
                if isinstance(result, dict):
                    citation = {
                        "title": result.get("title", "Untitled"),
                        "authors": result.get("authors", []),
                        "year": result.get("published_date", "")[:4] if result.get("published_date") else "",
                        "url": result.get("url", ""),
                        "doi": result.get("doi", ""),
                        "abstract": result.get("abstract", ""),
                        "source": source_type,
                        "id": self._generate_citation_id(result)
                    }
                    
                    # Clean up data
                    if not citation["year"] or not citation["year"].isdigit():
                        citation["year"] = "n.d."  # No date
                    
                    if not citation["authors"]:
                        citation["authors"] = ["Unknown"]
                    
                    citations.append(citation)
        
        return citations
    
    def _generate_citation_id(self, result: Dict[str, Any]) -> str:
        """
        Generate a citation ID for a search result
        
        Args:
            result: Search result
            
        Returns:
            Citation ID
        """
        # Get first author's last name or "unknown"
        if result.get("authors") and len(result["authors"]) > 0:
            first_author = result["authors"][0]
            # Extract last name
            if "," in first_author:
                last_name = first_author.split(",")[0].strip().lower()
            else:
                name_parts = first_author.split()
                last_name = name_parts[-1].lower() if name_parts else "unknown"
        else:
            last_name = "unknown"
        
        # Get year or "nd" for no date
        year = result.get("published_date", "")[:4] if result.get("published_date") else "nd"
        if not year.isdigit():
            year = "nd"
        
        # Get first word of title or "untitled"
        title = result.get("title", "untitled")
        first_word = title.split()[0].lower() if title else "untitled"
        # Remove non-alphanumeric characters
        first_word = ''.join(c for c in first_word if c.isalnum())
        
        # Combine to create citation ID
        citation_id = f"{last_name}{year}{first_word}"
        
        return citation_id
    
    def enhance_paper_with_citations(self, paper_content: str, topic: str, num_sources: int = 10) -> Dict[str, Any]:
        """
        Enhance a paper with citations and references
        
        Args:
            paper_content: Paper content to enhance
            topic: Paper topic
            num_sources: Number of sources to use
            
        Returns:
            Enhanced paper with citations and references
        """
        # Get citation data
        citation_data = self.get_citation_data(topic, num_sources)
        
        # Create citation instructions
        citation_instructions = f"""
You are an academic editor. Enhance the following paper by adding appropriate citations from the provided sources.
Every claim or statement should be supported by a citation where appropriate.

Paper Topic: {topic}

Available Sources for Citation:
"""
        
        # Add citation sources
        for i, citation in enumerate(citation_data, 1):
            authors = ", ".join(citation["authors"]) if citation["authors"] else "Unknown"
            citation_instructions += f"{i}. [{citation['id']}] {authors} ({citation['year']}). {citation['title']}.\n"
            if citation["abstract"]:
                abstract = citation["abstract"][:200] + "..." if len(citation["abstract"]) > 200 else citation["abstract"]
                citation_instructions += f"   Abstract: {abstract}\n"
            citation_instructions += "\n"
        
        citation_instructions += """
Instructions:
1. Add appropriate citations in (Author, Year) format.
2. Ensure every major claim has a citation.
3. Do not change the core content of the paper.
4. Add a complete References section at the end.
5. Ensure the paper is in English.

Paper Content:
"""
        
        # Combine instructions with paper content
        prompt = citation_instructions + paper_content
        
        # Use LLM to enhance paper with citations
        # This would typically call an LLM, but for now we'll return the data for the calling function to use
        return {
            "prompt": prompt,
            "citation_data": citation_data
        } 