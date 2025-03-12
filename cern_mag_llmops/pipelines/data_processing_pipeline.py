"""
Data processing pipeline for extracting text from PDFs and creating vector database
"""

import os
import logging
import argparse
import pandas as pd

from cern_mag_llmops.data_processing.pdf_processor import PDFProcessor
from cern_mag_llmops.data_processing.vector_store import VectorDatabaseBuilder
from cern_mag_llmops.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(settings.DATA_DIR, 'data_processing_pipeline.log'))
    ]
)
logger = logging.getLogger(__name__)


def run_data_processing(pdf_dir=None, output_dir=None, vectordb_dir=None, skip_pdf_processing=False):
    """
    Run the data processing pipeline
    
    Args:
        pdf_dir: Directory containing PDF files
        output_dir: Directory to save processed text files
        vectordb_dir: Directory to save vector database
        skip_pdf_processing: Skip PDF processing and use existing chunks
        
    Returns:
        dict: Summary of the processing
    """
    logger.info("Starting data processing pipeline")
    
    chunks_df = None
    chunks_path = os.path.join(
        output_dir or os.path.join(settings.DATA_DIR, "processed_text"),
        "processed_chunks.csv"
    )
    
    # Step 1: Process PDFs if not skipped
    if not skip_pdf_processing:
        logger.info("Step 1: Processing PDFs")
        processor = PDFProcessor(pdf_dir=pdf_dir, output_dir=output_dir)
        chunks_df = processor.process_pdfs()
        logger.info(f"Processed {len(chunks_df)} chunks from PDFs")
    else:
        logger.info("Skipping PDF processing, using existing chunks")
        if os.path.exists(chunks_path):
            chunks_df = pd.read_csv(chunks_path)
            logger.info(f"Loaded {len(chunks_df)} existing chunks")
        else:
            logger.error(f"No existing chunks found at {chunks_path}")
            raise FileNotFoundError(f"No existing chunks found at {chunks_path}")
    
    # Step 2: Create vector database
    logger.info("Step 2: Creating vector database")
    vector_builder = VectorDatabaseBuilder(
        chunks_path=chunks_path,
        db_dir=vectordb_dir
    )
    vectordb = vector_builder.build_database()
    logger.info("Vector database created successfully")
    
    # Return summary
    summary = {
        "num_chunks": len(chunks_df) if chunks_df is not None else 0,
        "chunks_path": chunks_path,
        "vectordb_dir": vectordb_dir or settings.VECTORDB_DIR
    }
    
    logger.info("Data processing pipeline completed")
    return summary


def main():
    """Main function to run the data processing pipeline from command line"""
    parser = argparse.ArgumentParser(description="CERN Magazine data processing pipeline")
    parser.add_argument("--pdf-dir", type=str, default=settings.PDFS_DIR,
                        help="Directory containing PDF files")
    parser.add_argument("--output-dir", type=str, default=os.path.join(settings.DATA_DIR, "processed_text"),
                        help="Directory to save processed text files")
    parser.add_argument("--vectordb-dir", type=str, default=settings.VECTORDB_DIR,
                        help="Directory to save vector database")
    parser.add_argument("--skip-pdf-processing", action="store_true",
                        help="Skip PDF processing and use existing chunks")
    
    args = parser.parse_args()
    
    run_data_processing(
        pdf_dir=args.pdf_dir,
        output_dir=args.output_dir,
        vectordb_dir=args.vectordb_dir,
        skip_pdf_processing=args.skip_pdf_processing
    )


if __name__ == "__main__":
    main() 