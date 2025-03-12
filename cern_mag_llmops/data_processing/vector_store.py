"""
Vector database creation and management for document retrieval
"""

import os
import logging
import pandas as pd
from tqdm import tqdm
import numpy as np
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
import time

from cern_mag_llmops.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(settings.DATA_DIR, 'vector_store.log'))
    ]
)
logger = logging.getLogger(__name__)


class VectorDatabaseBuilder:
    """
    Build and manage vector database for document retrieval
    """
    
    def __init__(self, chunks_path=None, db_dir=None):
        """
        Initialize the vector database builder
        
        Args:
            chunks_path: Path to the CSV file with processed chunks
            db_dir: Directory to save the vector database
        """
        self.chunks_path = chunks_path or os.path.join(
            settings.DATA_DIR, "processed_text", "processed_chunks.csv"
        )
        self.db_dir = db_dir or settings.VECTORDB_DIR
        
        # Create embeddings instance
        self.embeddings = OpenAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
        )
    
    def load_chunks(self):
        """
        Load processed text chunks from CSV
        
        Returns:
            pd.DataFrame: DataFrame with chunks
        """
        if not os.path.exists(self.chunks_path):
            raise FileNotFoundError(f"Chunks file not found: {self.chunks_path}")
        
        logger.info(f"Loading chunks from {self.chunks_path}")
        return pd.read_csv(self.chunks_path)
    
    def prepare_documents(self, chunks_df):
        """
        Prepare documents from DataFrame for vector storage
        
        Args:
            chunks_df: DataFrame with chunks
            
        Returns:
            list: List of Document objects
        """
        documents = []
        
        for _, row in chunks_df.iterrows():
            # Create a metadata dictionary with all fields except the text
            metadata = {col: str(row[col]) for col in row.index if col != 'text'}
            
            # Create a Document object
            doc = Document(
                page_content=row['text'],
                metadata=metadata
            )
            documents.append(doc)
        
        logger.info(f"Prepared {len(documents)} documents for vector database")
        return documents
    
    def create_vector_database(self, documents):
        """
        Create a vector database from documents
        
        Args:
            documents: List of Document objects
            
        Returns:
            Chroma: Vector database instance
        """
        logger.info(f"Creating vector database in {self.db_dir}")
        
        # Ensure the database directory exists
        os.makedirs(self.db_dir, exist_ok=True)
        
        # Clear existing database if it exists
        if os.path.exists(os.path.join(self.db_dir, "chroma.sqlite3")):
            logger.warning("Existing vector database found. It will be overwritten.")
        
        # Create batches of documents to avoid rate limits
        batch_size = settings.MAX_EMBEDDING_BATCH_SIZE
        document_batches = [documents[i:i + batch_size] for i in range(0, len(documents), batch_size)]
        
        # Create the database with the first batch
        vectordb = None
        
        for i, batch in enumerate(tqdm(document_batches, desc="Creating vector database")):
            try:
                if i == 0:
                    # Create new database with first batch
                    vectordb = Chroma.from_documents(
                        documents=batch,
                        embedding=self.embeddings,
                        persist_directory=self.db_dir
                    )
                else:
                    # Add to existing database
                    vectordb.add_documents(documents=batch)
                
                # Persist after each batch
                vectordb.persist()
                
                logger.info(f"Processed batch {i+1}/{len(document_batches)} with {len(batch)} documents")
                
                # Sleep to avoid rate limits
                if i < len(document_batches) - 1:
                    time.sleep(1)
                    
            except Exception as e:
                logger.error(f"Error processing batch {i+1}: {e}")
                # Continue with the next batch
                continue
        
        if vectordb:
            logger.info(f"Vector database created successfully with {len(documents)} documents")
            return vectordb
        else:
            raise RuntimeError("Failed to create vector database")
    
    def build_database(self):
        """
        Build the complete vector database
        
        Returns:
            Chroma: Vector database instance
        """
        try:
            # Load chunks
            chunks_df = self.load_chunks()
            
            # Prepare documents
            documents = self.prepare_documents(chunks_df)
            
            # Create vector database
            vectordb = self.create_vector_database(documents)
            
            # Test retrieval
            self.test_retrieval(vectordb)
            
            return vectordb
            
        except Exception as e:
            logger.error(f"Error building vector database: {e}")
            raise
    
    def test_retrieval(self, vectordb, query="CERN accelerator"):
        """
        Test retrieval from the vector database
        
        Args:
            vectordb: Vector database instance
            query: Test query
        """
        logger.info(f"Testing retrieval with query: '{query}'")
        
        retriever = vectordb.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
        
        try:
            results = retriever.get_relevant_documents(query)
            
            logger.info(f"Retrieved {len(results)} documents")
            for i, doc in enumerate(results):
                logger.info(f"Result {i+1}:")
                logger.info(f"  Source: {doc.metadata.get('filename', 'Unknown')}")
                logger.info(f"  Text: {doc.page_content[:100]}...")
                
        except Exception as e:
            logger.error(f"Error testing retrieval: {e}")


def main():
    """Main function to run the vector database builder independently"""
    builder = VectorDatabaseBuilder()
    builder.build_database()


if __name__ == "__main__":
    main() 