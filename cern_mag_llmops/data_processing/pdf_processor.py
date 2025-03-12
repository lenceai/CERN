"""
PDF processor for extracting and preprocessing text from CERN Courier PDFs
"""

import os
import glob
import logging
from pypdf import PdfReader
import re
from tqdm import tqdm
import pandas as pd

from cern_mag_llmops.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(settings.DATA_DIR, 'pdf_processor.log'))
    ]
)
logger = logging.getLogger(__name__)


class PDFProcessor:
    """
    Process PDF files to extract and preprocess text
    """
    
    def __init__(self, pdf_dir=None, output_dir=None):
        """
        Initialize the PDF processor
        
        Args:
            pdf_dir: Directory containing PDF files
            output_dir: Directory to save processed text files
        """
        self.pdf_dir = pdf_dir or settings.PDFS_DIR
        self.output_dir = output_dir or os.path.join(settings.DATA_DIR, "processed_text")
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def extract_text_from_pdf(self, pdf_path):
        """
        Extract text from a PDF file
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            str: Extracted text
        """
        try:
            reader = PdfReader(pdf_path)
            text = ""
            
            for i, page in enumerate(reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n\n"
                except Exception as e:
                    logger.warning(f"Error extracting text from page {i} of {pdf_path}: {e}")
            
            return text
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {e}")
            return ""
    
    def preprocess_text(self, text):
        """
        Preprocess extracted text
        
        Args:
            text: Text to preprocess
            
        Returns:
            str: Preprocessed text
        """
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters that don't add meaning
        text = re.sub(r'[^\w\s.,;:!?""\''\-–—()]', ' ', text)
        
        # Remove footer/header patterns often found in PDFs
        text = re.sub(r'(?i)page \d+ of \d+', '', text)
        text = re.sub(r'(?i)cern courier\s+[a-z]+\s+\d{4}', '', text)
        
        # Normalize whitespace again after replacements
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_metadata(self, pdf_path, text):
        """
        Extract metadata from PDF filename and content
        
        Args:
            pdf_path: Path to the PDF file
            text: Extracted text
            
        Returns:
            dict: Metadata including date, issue, title
        """
        filename = os.path.basename(pdf_path)
        
        # Try to extract year and month from filename
        date_match = re.search(r'(?i)(20\d{2})([a-z]+)', filename)
        year = date_match.group(1) if date_match else "Unknown"
        month = date_match.group(2) if date_match else "Unknown"
        
        # Extract issue information if present
        issue_match = re.search(r'(?i)issue\s*(\d+)', filename)
        issue = issue_match.group(1) if issue_match else "Unknown"
        
        # Extract volume if present
        volume_match = re.search(r'(?i)volume\s*(\d+)', filename)
        volume = volume_match.group(1) if volume_match else "Unknown"
        
        return {
            "filename": filename,
            "year": year,
            "month": month,
            "issue": issue,
            "volume": volume,
            "file_path": pdf_path
        }
    
    def chunk_text(self, text, metadata, chunk_size=None, overlap=None):
        """
        Split text into overlapping chunks for processing
        
        Args:
            text: Text to chunk
            metadata: Metadata to include with each chunk
            chunk_size: Size of each chunk
            overlap: Overlap between chunks
            
        Returns:
            list: List of dictionaries with chunked text and metadata
        """
        if not text:
            return []
        
        chunk_size = chunk_size or settings.CHUNK_SIZE
        overlap = overlap or settings.CHUNK_OVERLAP
        
        # Split by sentences to avoid cutting in the middle of a sentence
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # If adding this sentence would exceed the chunk size and we have content
            if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
                # Save the current chunk
                chunk_data = metadata.copy()
                chunk_data["text"] = current_chunk
                chunk_data["chunk_len"] = len(current_chunk)
                chunks.append(chunk_data)
                
                # Start a new chunk, keeping the overlap
                words = current_chunk.split()
                overlap_words = words[-min(len(words), overlap // 5):]  # Approximating words to chars
                current_chunk = " ".join(overlap_words) + " " + sentence
            else:
                # Add the sentence to the current chunk
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
        
        # Add the last chunk if there's content
        if current_chunk:
            chunk_data = metadata.copy()
            chunk_data["text"] = current_chunk
            chunk_data["chunk_len"] = len(current_chunk)
            chunks.append(chunk_data)
        
        return chunks
    
    def process_pdfs(self):
        """
        Process all PDFs in the directory
        
        Returns:
            pd.DataFrame: DataFrame with processed text chunks and metadata
        """
        pdf_files = glob.glob(os.path.join(self.pdf_dir, "*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        all_chunks = []
        
        for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
            logger.info(f"Processing {pdf_path}")
            
            # Extract text from PDF
            extracted_text = self.extract_text_from_pdf(pdf_path)
            
            if not extracted_text:
                logger.warning(f"No text extracted from {pdf_path}")
                continue
            
            # Preprocess the extracted text
            processed_text = self.preprocess_text(extracted_text)
            
            if not processed_text:
                logger.warning(f"No text remaining after preprocessing {pdf_path}")
                continue
                
            # Extract metadata
            metadata = self.extract_metadata(pdf_path, processed_text)
            
            # Chunk the text
            chunks = self.chunk_text(processed_text, metadata)
            
            logger.info(f"Created {len(chunks)} chunks from {pdf_path}")
            all_chunks.extend(chunks)
        
        # Create DataFrame
        chunks_df = pd.DataFrame(all_chunks)
        
        # Save to CSV
        csv_path = os.path.join(self.output_dir, "processed_chunks.csv")
        chunks_df.to_csv(csv_path, index=False)
        logger.info(f"Saved {len(chunks_df)} processed chunks to {csv_path}")
        
        return chunks_df


def main():
    """Main function to run the PDF processor independently"""
    processor = PDFProcessor()
    processor.process_pdfs()


if __name__ == "__main__":
    main() 