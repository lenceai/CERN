"""
CERN Courier PDF crawler for downloading magazine archives
"""

import requests
from bs4 import BeautifulSoup
import os
import time
from urllib.parse import urljoin, unquote
import re
from tqdm import tqdm
import logging

from cern_mag_llmops.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(settings.DATA_DIR, 'crawler.log'))
    ]
)
logger = logging.getLogger(__name__)

class CERNPDFCrawler:
    """
    Crawler for downloading CERN Courier PDF magazines
    """
    
    def __init__(self, download_folder=None):
        """
        Initialize the CERN PDF crawler
        
        Args:
            download_folder: Directory to save the downloaded PDFs
        """
        self.base_url = settings.CERN_BASE_URL
        self.session = requests.Session()
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.download_folder = download_folder or settings.PDFS_DIR
        self.processed_article_urls = set()
        self.downloaded_files = set()
        
        if not os.path.exists(self.download_folder):
            os.makedirs(self.download_folder)
        self.load_existing_files()

    def load_existing_files(self):
        """Load existing PDF files to avoid re-downloading"""
        for filename in os.listdir(self.download_folder):
            if filename.lower().endswith('.pdf'):
                self.downloaded_files.add(filename)
        logger.info(f"Found {len(self.downloaded_files)} existing PDF files")

    def get_page_content(self, url):
        """
        Get the HTML content of a page with retries
        
        Args:
            url: URL to fetch
            
        Returns:
            str: HTML content of the page or None if failed
        """
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.session.get(url, headers=self.headers)
                response.raise_for_status()
                return response.text
            except requests.RequestException as e:
                if attempt == max_retries - 1:
                    logger.error(f"Error fetching {url}: {e}")
                    return None
                time.sleep(2 ** attempt)
        return None

    def extract_pdf_urls_from_text(self, text):
        """
        Extract PDF URLs from text content including 'File path:' patterns
        
        Args:
            text: HTML or text content to search for PDF URLs
            
        Returns:
            set: Set of PDF URLs found
        """
        pdf_urls = set()
        
        # Look for "File path:" pattern
        file_path_matches = re.finditer(r'File path:\s*(https?://[^\s<>"]+\.pdf)', text, re.IGNORECASE)
        for match in file_path_matches:
            pdf_urls.add(match.group(1))
            
        # Look for direct PDF links
        pdf_link_matches = re.finditer(r'href="(https?://[^\s<>"]+\.pdf)"', text, re.IGNORECASE)
        for match in pdf_link_matches:
            pdf_urls.add(match.group(1))
            
        return pdf_urls

    def find_courier_links(self, page_url):
        """
        Find CERN Courier article links on a page
        
        Args:
            page_url: URL of the page to search
            
        Returns:
            list: List of article URLs
        """
        content = self.get_page_content(page_url)
        if not content:
            return []
        
        soup = BeautifulSoup(content, 'html.parser')
        courier_links = []
        
        for link in soup.find_all('a', href=True):
            href = link['href']
            if '/resources/courier/' in href or '/record/' in href:
                full_url = urljoin("https://home.cern", href)
                if full_url not in self.processed_article_urls:
                    courier_links.append(full_url)
                    self.processed_article_urls.add(full_url)
        
        return courier_links

    def find_pdf_links(self, article_url):
        """
        Find all PDF download links on an article page
        
        Args:
            article_url: URL of the article page
            
        Returns:
            list: List of PDF URLs
        """
        content = self.get_page_content(article_url)
        if not content:
            return []
        
        pdf_urls = set()
        
        # Extract URLs from text content
        pdf_urls.update(self.extract_pdf_urls_from_text(content))
        
        # Parse with BeautifulSoup for structured extraction
        soup = BeautifulSoup(content, 'html.parser')
        
        # Look for links containing PDF
        for link in soup.find_all('a', href=True):
            href = link['href']
            if href.lower().endswith('.pdf'):
                full_url = urljoin("https://home.cern", href)
                pdf_urls.add(full_url)
        
        return list(pdf_urls)

    def sanitize_filename(self, url):
        """
        Create a safe filename from URL
        
        Args:
            url: URL of the PDF
            
        Returns:
            str: Sanitized filename
        """
        filename = unquote(url.split('/')[-1])
        # Remove or replace unsafe characters
        filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
        return filename

    def download_pdf(self, pdf_url, filename):
        """
        Download a PDF file
        
        Args:
            pdf_url: URL of the PDF to download
            filename: Filename to save as
            
        Returns:
            bool: True if successful, False otherwise
        """
        if filename in self.downloaded_files:
            logger.info(f"Skipping {filename} - already downloaded")
            return True
            
        try:
            response = self.session.get(pdf_url, headers=self.headers, stream=True)
            response.raise_for_status()
            
            file_path = os.path.join(self.download_folder, filename)
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(file_path, 'wb') as file, tqdm(
                desc=filename,
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for data in response.iter_content(chunk_size=1024):
                    size = file.write(data)
                    pbar.update(size)
            
            self.downloaded_files.add(filename)
            logger.info(f"Successfully downloaded {filename}")
            return True
        except Exception as e:
            logger.error(f"Error downloading {filename}: {e}")
            return False

    def crawl_and_download(self, start_page=None, end_page=None):
        """
        Crawl CERN Courier pages and download PDF files
        
        Args:
            start_page: Starting page number
            end_page: Ending page number
            
        Returns:
            dict: Summary of the crawl and download process
        """
        start_page = start_page if start_page is not None else settings.CRAWL_START_PAGE
        end_page = end_page if end_page is not None else settings.CRAWL_END_PAGE
        
        logger.info(f"Starting CERN PDF crawler (pages {start_page} to {end_page})")
        
        found_pdfs = 0
        downloaded_pdfs = 0
        skipped_pdfs = 0
        failed_downloads = []
        
        for page_num in range(start_page, end_page + 1):
            page_url = f"{self.base_url}?type=52&page={page_num}"
            logger.info(f"Processing page {page_num}...")
            
            courier_links = self.find_courier_links(page_url)
            logger.info(f"Found {len(courier_links)} new article links on page {page_num}")
            
            for article_url in courier_links:
                pdf_urls = self.find_pdf_links(article_url)
                
                for pdf_url in pdf_urls:
                    filename = self.sanitize_filename(pdf_url)
                    found_pdfs += 1
                    
                    logger.info(f"Found PDF: {filename}")
                    logger.info(f"URL: {pdf_url}")
                    
                    if self.download_pdf(pdf_url, filename):
                        if filename in self.downloaded_files:
                            skipped_pdfs += 1
                        else:
                            downloaded_pdfs += 1
                    else:
                        failed_downloads.append(filename)
                
                # Add delay between requests to avoid overloading the server
                time.sleep(settings.CRAWL_DELAY)
        
        # Prepare summary
        summary = {
            "total_found": found_pdfs,
            "downloaded": downloaded_pdfs,
            "skipped": skipped_pdfs,
            "failed": len(failed_downloads),
            "failed_files": failed_downloads
        }
        
        # Log summary
        logger.info("\nDownload Summary:")
        logger.info("--------------------")
        logger.info(f"Total PDFs found: {found_pdfs}")
        logger.info(f"Successfully downloaded: {downloaded_pdfs}")
        logger.info(f"Skipped (already downloaded): {skipped_pdfs}")
        logger.info(f"Failed downloads: {len(failed_downloads)}")
        
        if failed_downloads:
            logger.info("\nFailed downloads:")
            for filename in failed_downloads:
                logger.info(f"- {filename}")
        
        return summary


def main():
    """Main function to run the crawler independently"""
    crawler = CERNPDFCrawler()
    crawler.crawl_and_download()


if __name__ == "__main__":
    main() 