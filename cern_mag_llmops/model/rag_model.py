"""
RAG (Retrieval Augmented Generation) model for CERN Magazine Q&A
"""

import os
import logging
import time
from typing import List, Dict, Any, Optional

from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

from cern_mag_llmops.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(settings.DATA_DIR, 'rag_model.log'))
    ]
)
logger = logging.getLogger(__name__)


class RAGModel:
    """
    RAG model for CERN Magazine Q&A
    """
    
    def __init__(self, vectordb_dir=None):
        """
        Initialize the RAG model
        
        Args:
            vectordb_dir: Directory containing the vector database
        """
        self.vectordb_dir = vectordb_dir or settings.VECTORDB_DIR
        
        # Initialize components
        self._init_components()
    
    def _init_components(self):
        """Initialize LLM, vector store, retriever, and prompt template"""
        try:
            # Initialize vector store
            self.vectorstore = Chroma(
                persist_directory=self.vectordb_dir,
                embedding_function=OpenAIEmbeddings(model=settings.EMBEDDING_MODEL)
            )
            
            # Initialize LLM
            self.llm = ChatOpenAI(
                model=settings.CHAT_MODEL,
                temperature=0
            )
            
            # Initialize retriever
            self.retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": settings.TOP_K_DOCUMENTS}
            )
            
            # Define prompt template
            template = """You are a helpful research assistant with access to CERN Courier articles.
            Use the following articles to answer the question. If you can't answer the question based
            on the articles, say so clearly.

            Context articles:
            {context}

            Question: {question}

            Please provide a detailed answer with specific references to the articles when possible:"""
            
            self.prompt = ChatPromptTemplate.from_template(template)
            
            # Define RAG chain
            self.rag_chain = (
                RunnableParallel(
                    {"context": self.retriever, "question": RunnablePassthrough()}
                )
                | self.prompt
                | self.llm
                | StrOutputParser()
            )
            
            logger.info("RAG model components initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing RAG model components: {e}")
            raise
    
    def answer_question(self, question: str) -> Dict[str, Any]:
        """
        Answer a question using the RAG model
        
        Args:
            question: User question
            
        Returns:
            dict: Answer and metadata
        """
        start_time = time.time()
        
        try:
            # Get answer from RAG chain
            answer = self.rag_chain.invoke(question)
            
            # Get retrieved documents for transparency
            docs = self.retriever.get_relevant_documents(question)
            sources = [
                {
                    "filename": doc.metadata.get("filename", "Unknown"),
                    "year": doc.metadata.get("year", "Unknown"),
                    "issue": doc.metadata.get("issue", "Unknown"),
                    "excerpt": doc.page_content[:200] + "..."
                }
                for doc in docs
            ]
            
            end_time = time.time()
            response_time = round(end_time - start_time, 2)
            
            result = {
                "question": question,
                "answer": answer,
                "sources": sources,
                "response_time": response_time,
                "model": settings.CHAT_MODEL,
                "num_sources": len(sources)
            }
            
            logger.info(f"Answered question in {response_time} seconds using {len(sources)} sources")
            return result
            
        except Exception as e:
            logger.error(f"Error answering question: {e}")
            end_time = time.time()
            response_time = round(end_time - start_time, 2)
            
            return {
                "question": question,
                "answer": f"I'm sorry, I encountered an error while trying to answer your question: {str(e)}",
                "sources": [],
                "response_time": response_time,
                "model": settings.CHAT_MODEL,
                "num_sources": 0,
                "error": str(e)
            }
    
    def get_relevant_documents(self, query: str, k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get relevant documents for a query
        
        Args:
            query: Search query
            k: Number of documents to retrieve
            
        Returns:
            list: List of document dictionaries
        """
        try:
            k = k or settings.TOP_K_DOCUMENTS
            docs = self.retriever.get_relevant_documents(query)
            
            # Convert to dictionaries
            doc_dicts = [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in docs
            ]
            
            return doc_dicts
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return []


def main():
    """Test the RAG model with a sample question"""
    rag_model = RAGModel()
    sample_question = "What is the significance of the Higgs boson discovery?"
    
    print(f"Testing RAG model with question: '{sample_question}'")
    result = rag_model.answer_question(sample_question)
    
    print("\nAnswer:")
    print(result["answer"])
    
    print("\nSources:")
    for i, source in enumerate(result["sources"]):
        print(f"{i+1}. {source['filename']} ({source['year']})")
    
    print(f"\nResponse time: {result['response_time']} seconds")


if __name__ == "__main__":
    main() 