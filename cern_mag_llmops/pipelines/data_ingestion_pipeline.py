"""
Data ingestion pipeline for downloading CERN Courier PDFs
"""

import os
import logging
import argparse
from cern_mag_llmops.data_ingestion.pdf_crawler import CERNPDFCrawler
from cern_mag_llmops.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(settings.DATA_DIR, 'data_ingestion_pipeline.log'))
    ]
)
logger = logging.getLogger(__name__)


def run_data_ingestion(start_page=None, end_page=None, download_folder=None):
    """
    Run the data ingestion pipeline
    
    Args:
        start_page: Starting page number for crawling
        end_page: Ending page number for crawling
        download_folder: Directory to save downloaded PDFs
        
    Returns:
        dict: Summary of the crawl and download process
    """
    logger.info("Starting data ingestion pipeline")
    
    # Initialize crawler
    crawler = CERNPDFCrawler(download_folder=download_folder)
    
    # Run crawler
    summary = crawler.crawl_and_download(
        start_page=start_page,
        end_page=end_page
    )
    
    logger.info("Data ingestion pipeline completed")
    logger.info(f"Total PDFs found: {summary['total_found']}")
    logger.info(f"Successfully downloaded: {summary['downloaded']}")
    logger.info(f"Skipped (already downloaded): {summary['skipped']}")
    logger.info(f"Failed downloads: {summary['failed']}")
    
    return summary


def main():
    """Main function to run the data ingestion pipeline from command line"""
    parser = argparse.ArgumentParser(description="CERN Magazine data ingestion pipeline")
    parser.add_argument("--start-page", type=int, default=settings.CRAWL_START_PAGE,
                        help="Starting page number for crawling")
    parser.add_argument("--end-page", type=int, default=settings.CRAWL_END_PAGE,
                        help="Ending page number for crawling")
    parser.add_argument("--download-folder", type=str, default=settings.PDFS_DIR,
                        help="Directory to save downloaded PDFs")
    
    args = parser.parse_args()
    
    run_data_ingestion(
        start_page=args.start_page,
        end_page=args.end_page,
        download_folder=args.download_folder
    )


if __name__ == "__main__":
    main() 