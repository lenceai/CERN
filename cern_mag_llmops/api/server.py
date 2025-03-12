"""
FastAPI server for CERN Magazine LLMOps framework
"""

import os
import time
import json
import logging
from typing import Dict, Any, List, Optional

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from cern_mag_llmops.config import settings
from cern_mag_llmops.model.rag_model import RAGModel
from cern_mag_llmops.model.model_comparison import ModelComparison
from cern_mag_llmops.api.models import (
    QueryRequest, 
    QueryResponse,
    DocumentRequest,
    DocumentResponse,
    Document,
    DocumentMetadata,
    ModelResponse,
    Source,
    HealthResponse
)
from cern_mag_llmops import __version__

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(settings.DATA_DIR, 'api_server.log'))
    ]
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="CERN Magazine Research Assistant API",
    description="API for querying the CERN Magazine Research Assistant",
    version=__version__
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instances
rag_model = None
model_comparison = None

def get_rag_model() -> RAGModel:
    """Get or initialize the RAG model"""
    global rag_model
    if rag_model is None:
        try:
            logger.info("Initializing RAG model")
            rag_model = RAGModel()
        except Exception as e:
            logger.error(f"Error initializing RAG model: {e}")
            raise HTTPException(status_code=500, detail=f"Error initializing RAG model: {str(e)}")
    return rag_model

def get_model_comparison() -> ModelComparison:
    """Get or initialize the model comparison"""
    global model_comparison
    if model_comparison is None:
        try:
            logger.info("Initializing model comparison")
            model_comparison = ModelComparison()
        except Exception as e:
            logger.error(f"Error initializing model comparison: {e}")
            raise HTTPException(status_code=500, detail=f"Error initializing model comparison: {str(e)}")
    return model_comparison


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "CERN Magazine Research Assistant API",
        "docs_url": "/docs",
        "version": __version__
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    # Check if models are available
    models_available = {
        "rag": True
    }
    
    try:
        # Try to initialize RAG model
        rag_model = get_rag_model()
    except Exception:
        models_available["rag"] = False
    
    # Check if fine-tuned model is available
    fine_tuned_model_id = settings.FINE_TUNED_MODEL
    models_available["fine_tuned"] = bool(fine_tuned_model_id)
    
    return HealthResponse(
        status="ok",
        version=__version__,
        models=models_available,
        timestamp=time.time()
    )


@app.post("/query", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    rag_model: RAGModel = Depends(get_rag_model),
    model_comparison: ModelComparison = Depends(get_model_comparison)
):
    """
    Answer a question using the specified model type
    
    - **rag**: Use the RAG (Retrieval Augmented Generation) model
    - **fine_tuned**: Use the fine-tuned model (if available)
    - **compare**: Compare both models (if fine-tuned model is available)
    """
    try:
        model_type = request.model_type.lower()
        logger.info(f"Query request with model_type: {model_type}, question: {request.question}")
        
        # Check if model type is valid
        if model_type not in ["rag", "fine_tuned", "compare"]:
            raise HTTPException(status_code=400, detail=f"Invalid model_type: {model_type}")
        
        # Check if fine-tuned model is available when requested
        if model_type in ["fine_tuned", "compare"] and not settings.FINE_TUNED_MODEL:
            logger.warning("Fine-tuned model requested but not available")
            if model_type == "fine_tuned":
                raise HTTPException(status_code=400, detail="Fine-tuned model not available")
            else:
                # Fall back to RAG for comparison
                model_type = "rag"
        
        # Process based on model type
        if model_type == "rag":
            # Use RAG model
            rag_result = rag_model.answer_question(request.question)
            
            # Convert sources to Source objects
            sources = None
            if request.include_sources and "sources" in rag_result:
                sources = [
                    Source(
                        filename=source.get("filename", "Unknown"),
                        year=source.get("year", "Unknown"),
                        issue=source.get("issue", "Unknown"),
                        excerpt=source.get("excerpt", "")
                    )
                    for source in rag_result.get("sources", [])[:request.max_sources]
                ]
            
            # Create response
            return QueryResponse(
                question=request.question,
                rag=ModelResponse(
                    answer=rag_result["answer"],
                    model=rag_result.get("model", "Unknown"),
                    response_time=rag_result.get("response_time", 0),
                    sources=sources
                ),
                fine_tuned=None,
                model_type=model_type
            )
            
        elif model_type == "fine_tuned":
            # Use fine-tuned model
            ft_result = model_comparison.query_fine_tuned_model(request.question)
            
            # Create response
            return QueryResponse(
                question=request.question,
                rag=None,
                fine_tuned=ModelResponse(
                    answer=ft_result["answer"],
                    model=ft_result.get("model", settings.FINE_TUNED_MODEL),
                    response_time=ft_result.get("response_time", 0),
                    sources=None
                ),
                model_type=model_type
            )
            
        else:  # compare
            # Compare both models
            comparison_result = model_comparison.compare_models(request.question)
            
            # Convert sources to Source objects
            sources = None
            if request.include_sources and "sources" in comparison_result["rag"]:
                sources = [
                    Source(
                        filename=source.get("filename", "Unknown"),
                        year=source.get("year", "Unknown"),
                        issue=source.get("issue", "Unknown"),
                        excerpt=source.get("excerpt", "")
                    )
                    for source in comparison_result["rag"].get("sources", [])[:request.max_sources]
                ]
            
            # Create response
            return QueryResponse(
                question=request.question,
                rag=ModelResponse(
                    answer=comparison_result["rag"]["answer"],
                    model=comparison_result["rag"].get("model", "Unknown"),
                    response_time=comparison_result["rag"].get("response_time", 0),
                    sources=sources
                ),
                fine_tuned=ModelResponse(
                    answer=comparison_result["fine_tuned"]["answer"],
                    model=comparison_result["fine_tuned"].get("model", settings.FINE_TUNED_MODEL),
                    response_time=comparison_result["fine_tuned"].get("response_time", 0),
                    sources=None
                ),
                model_type=model_type
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.post("/documents", response_model=DocumentResponse)
async def retrieve_documents(
    request: DocumentRequest,
    rag_model: RAGModel = Depends(get_rag_model)
):
    """
    Retrieve relevant documents for a query
    """
    try:
        logger.info(f"Document retrieval request: {request.query}")
        
        # Get relevant documents
        docs = rag_model.get_relevant_documents(request.query, request.k)
        
        # Convert to Document objects
        documents = []
        for doc in docs:
            metadata_dict = doc.get("metadata", {})
            metadata = DocumentMetadata(
                filename=metadata_dict.get("filename", "Unknown"),
                year=metadata_dict.get("year", "Unknown"),
                issue=metadata_dict.get("issue", "Unknown"),
                volume=metadata_dict.get("volume", "Unknown"),
                chunk_len=metadata_dict.get("chunk_len", 0)
            )
            
            document = Document(
                content=doc["content"],
                metadata=metadata
            )
            documents.append(document)
        
        return DocumentResponse(
            query=request.query,
            documents=documents,
            count=len(documents)
        )
        
    except Exception as e:
        logger.error(f"Error retrieving documents: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving documents: {str(e)}")


def start():
    """Start the API server"""
    uvicorn.run(
        "cern_mag_llmops.api.server:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG_MODE
    )


if __name__ == "__main__":
    start() 