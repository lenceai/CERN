"""
Pydantic models for API request and response validation
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request model for question answering"""
    question: str = Field(..., description="The question to answer about CERN research")
    model_type: Optional[str] = Field("rag", description="Model type to use: 'rag', 'fine_tuned', or 'compare'")
    include_sources: Optional[bool] = Field(True, description="Whether to include source documents in the response")
    max_sources: Optional[int] = Field(4, description="Maximum number of sources to include in the response")


class Source(BaseModel):
    """Source document information"""
    filename: str = Field(..., description="Filename of the source document")
    year: Optional[str] = Field(None, description="Publication year of the source")
    issue: Optional[str] = Field(None, description="Issue number of the source")
    excerpt: Optional[str] = Field(None, description="Short excerpt from the source")


class ModelResponse(BaseModel):
    """Response from a single model"""
    answer: str = Field(..., description="The answer to the question")
    model: str = Field(..., description="The model used to generate the answer")
    response_time: float = Field(..., description="Response time in seconds")
    sources: Optional[List[Source]] = Field(None, description="Source documents used to generate the answer")


class QueryResponse(BaseModel):
    """Response model for question answering"""
    question: str = Field(..., description="The original question")
    rag: Optional[ModelResponse] = Field(None, description="Response from the RAG model")
    fine_tuned: Optional[ModelResponse] = Field(None, description="Response from the fine-tuned model")
    model_type: str = Field(..., description="The model type that was used")


class DocumentRequest(BaseModel):
    """Request model for document retrieval"""
    query: str = Field(..., description="The search query")
    k: Optional[int] = Field(4, description="Number of documents to retrieve")


class DocumentMetadata(BaseModel):
    """Document metadata"""
    filename: Optional[str] = Field(None, description="Filename of the document")
    year: Optional[str] = Field(None, description="Publication year")
    issue: Optional[str] = Field(None, description="Issue number")
    volume: Optional[str] = Field(None, description="Volume number")
    chunk_len: Optional[int] = Field(None, description="Length of the chunk")


class Document(BaseModel):
    """Document model"""
    content: str = Field(..., description="Document content")
    metadata: DocumentMetadata = Field(..., description="Document metadata")


class DocumentResponse(BaseModel):
    """Response model for document retrieval"""
    query: str = Field(..., description="The original query")
    documents: List[Document] = Field(..., description="Retrieved documents")
    count: int = Field(..., description="Number of documents retrieved")


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    models: Dict[str, bool] = Field(..., description="Available models")
    timestamp: float = Field(..., description="Current timestamp") 