from __future__ import annotations

import os
import re
from typing import List, Dict, Any
from dataclasses import dataclass

from ..utils.logging_utils import logger, timeit


@dataclass
class DocumentChunk:
    """A chunk of text from a larger document."""
    content: str
    chunk_id: int
    source_info: Dict[str, Any]
    keywords: List[str]
    start_pos: int
    end_pos: int


class SimpleRAG:
    """Simple RAG implementation for SEC filings without requiring vector databases."""
    
    def __init__(self, chunk_size: int = 2000, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap
        
    def chunk_document(self, text: str, source_info: Dict[str, Any]) -> List[DocumentChunk]:
        """Split document into overlapping chunks while preserving table boundaries."""
        chunks = []
        
        # First, identify table boundaries
        table_starts = [m.start() for m in re.finditer(r"=== TABLE START ===", text)]
        table_ends = [m.end() for m in re.finditer(r"=== TABLE END ===", text)]
        
        # Create chunks, being careful not to split tables
        start_pos = 0
        chunk_id = 0
        
        while start_pos < len(text):
            end_pos = start_pos + self.chunk_size
            
            # Check if we would split a table
            if end_pos < len(text):
                # Look for table boundaries near the end position
                for table_start in table_starts:
                    if start_pos < table_start < end_pos:
                        # If there's a table starting in this chunk, include the whole table
                        for table_end in table_ends:
                            if table_end > table_start:
                                end_pos = max(end_pos, table_end)
                                break
                
                # Try to end at paragraph boundary
                last_para = text.rfind("\n\n", start_pos, end_pos)
                if last_para > start_pos + self.chunk_size // 2:
                    end_pos = last_para
            
            chunk_text = text[start_pos:end_pos].strip()
            if chunk_text:
                # Extract keywords for this chunk
                keywords = self._extract_keywords(chunk_text)
                
                chunks.append(DocumentChunk(
                    content=chunk_text,
                    chunk_id=chunk_id,
                    source_info=source_info,
                    keywords=keywords,
                    start_pos=start_pos,
                    end_pos=end_pos
                ))
                chunk_id += 1
            
            # Move to next chunk with overlap
            start_pos = max(start_pos + self.chunk_size - self.overlap, end_pos)
            
        return chunks
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text chunk for better retrieval."""
        # Common financial keywords
        financial_keywords = [
            "compensation", "salary", "bonus", "stock", "options", "awards",
            "ceo", "chief executive", "executive compensation", "summary compensation",
            "total compensation", "annual", "incentive", "equity", "non-equity"
        ]
        
        text_lower = text.lower()
        found_keywords = []
        
        for keyword in financial_keywords:
            if keyword in text_lower:
                found_keywords.append(keyword)
                
        # Also look for dollar amounts
        if re.search(r"\$[\d,]+", text):
            found_keywords.append("dollar_amounts")
            
        # Look for table indicators
        if "TABLE START" in text or "|" in text:
            found_keywords.append("table_data")
            
        return found_keywords
    
    @timeit 
    def search_relevant_chunks(self, chunks: List[DocumentChunk], query: str, max_chunks: int = 5) -> List[DocumentChunk]:
        """Search for chunks most relevant to the query."""
        query_lower = query.lower()
        query_terms = [term.strip() for term in re.split(r'[,\s]+', query_lower) if term.strip()]
        
        # Score chunks based on relevance
        scored_chunks = []
        
        for chunk in chunks:
            score = 0
            content_lower = chunk.content.lower()
            
            # Keyword matching
            for keyword in chunk.keywords:
                for term in query_terms:
                    if term in keyword or keyword in term:
                        score += 2
            
            # Direct text matching
            for term in query_terms:
                score += content_lower.count(term)
            
            # Bonus for tables when looking for numerical data  
            if "table_data" in chunk.keywords and any(term in ["compensation", "salary", "ceo"] for term in query_terms):
                score += 5
                
            # Bonus for dollar amounts when looking for compensation
            if "dollar_amounts" in chunk.keywords and any(term in ["compensation", "salary", "total"] for term in query_terms):
                score += 3
            
            if score > 0:
                scored_chunks.append((score, chunk))
        
        # Sort by score and return top chunks
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        return [chunk for _, chunk in scored_chunks[:max_chunks]]
    
    @timeit
    def extract_context_for_query(self, text: str, source_info: Dict[str, Any], query: str) -> str:
        """Extract the most relevant context from a document for a specific query."""
        # Chunk the document
        chunks = self.chunk_document(text, source_info)
        logger.info(f"Document chunked into {len(chunks)} chunks")
        
        # Find most relevant chunks
        relevant_chunks = self.search_relevant_chunks(chunks, query, max_chunks=3)
        logger.info(f"Found {len(relevant_chunks)} relevant chunks for query: {query}")
        
        # Combine relevant chunks
        if not relevant_chunks:
            # Fallback to first few chunks if no good matches
            relevant_chunks = chunks[:2]
            logger.warning("No highly relevant chunks found, using document start")
        
        context_parts = []
        for i, chunk in enumerate(relevant_chunks):
            context_parts.append(f"\n--- Relevant Section {i+1} ---\n{chunk.content}")
        
        return "\n".join(context_parts)