"""
Command-line interface for the CERN Magazine LLMOps framework
"""

import os
import sys
import argparse
import logging
from typing import List, Optional

from cern_mag_llmops.config import settings
from cern_mag_llmops.pipelines.data_ingestion_pipeline import run_data_ingestion
from cern_mag_llmops.pipelines.data_processing_pipeline import run_data_processing
from cern_mag_llmops.pipelines.fine_tuning_pipeline import run_fine_tuning
from cern_mag_llmops.model.model_comparison import ModelComparison
from cern_mag_llmops.api.server import start as start_api_server

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(settings.DATA_DIR, 'cli.log'))
    ]
)
logger = logging.getLogger(__name__)


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """
    Parse command-line arguments
    
    Args:
        args: Command-line arguments
        
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="CERN Magazine LLMOps CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run the data ingestion pipeline
  python -m cern_mag_llmops.utils.cli ingest
  
  # Run the data processing pipeline
  python -m cern_mag_llmops.utils.cli process
  
  # Run the fine-tuning pipeline
  python -m cern_mag_llmops.utils.cli finetune
  
  # Start the API server
  python -m cern_mag_llmops.utils.cli serve
  
  # Run the interactive model comparison
  python -m cern_mag_llmops.utils.cli compare
  
  # Run the full pipeline (ingest, process, serve)
  python -m cern_mag_llmops.utils.cli full
"""
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Run the data ingestion pipeline")
    ingest_parser.add_argument("--start-page", type=int, default=settings.CRAWL_START_PAGE,
                              help="Starting page number for crawling")
    ingest_parser.add_argument("--end-page", type=int, default=settings.CRAWL_END_PAGE,
                              help="Ending page number for crawling")
    
    # Process command
    process_parser = subparsers.add_parser("process", help="Run the data processing pipeline")
    process_parser.add_argument("--skip-pdf-processing", action="store_true",
                               help="Skip PDF processing and use existing chunks")
    
    # Finetune command
    finetune_parser = subparsers.add_parser("finetune", help="Run the fine-tuning pipeline")
    finetune_parser.add_argument("--num-qa-pairs", type=int, default=100,
                                help="Number of QA pairs to generate")
    finetune_parser.add_argument("--base-model", type=str, default="gpt-3.5-turbo",
                                help="Base model to fine-tune")
    
    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Start the API server")
    serve_parser.add_argument("--host", type=str, default=settings.API_HOST,
                             help="Host to bind the server to")
    serve_parser.add_argument("--port", type=int, default=settings.API_PORT,
                             help="Port to bind the server to")
    serve_parser.add_argument("--debug", action="store_true",
                             help="Run the server in debug mode")
    
    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Run the interactive model comparison")
    
    # Full command
    full_parser = subparsers.add_parser("full", help="Run the full pipeline (ingest, process, serve)")
    full_parser.add_argument("--skip-ingest", action="store_true",
                            help="Skip the data ingestion step")
    full_parser.add_argument("--skip-process", action="store_true",
                            help="Skip the data processing step")
    
    return parser.parse_args(args)


def main(args: Optional[List[str]] = None) -> None:
    """
    Main CLI function
    
    Args:
        args: Command-line arguments
    """
    parsed_args = parse_args(args)
    
    if not parsed_args.command:
        print("No command specified. Use --help for usage information.")
        sys.exit(1)
    
    try:
        if parsed_args.command == "ingest":
            logger.info("Running data ingestion pipeline")
            run_data_ingestion(
                start_page=parsed_args.start_page,
                end_page=parsed_args.end_page
            )
            
        elif parsed_args.command == "process":
            logger.info("Running data processing pipeline")
            run_data_processing(
                skip_pdf_processing=parsed_args.skip_pdf_processing
            )
            
        elif parsed_args.command == "finetune":
            logger.info("Running fine-tuning pipeline")
            run_fine_tuning(
                num_qa_pairs=parsed_args.num_qa_pairs,
                base_model=parsed_args.base_model
            )
            
        elif parsed_args.command == "serve":
            logger.info("Starting API server")
            # Override settings with command-line arguments
            settings.API_HOST = parsed_args.host
            settings.API_PORT = parsed_args.port
            settings.DEBUG_MODE = parsed_args.debug
            start_api_server()
            
        elif parsed_args.command == "compare":
            logger.info("Running interactive model comparison")
            comparison = ModelComparison()
            comparison.interactive_comparison()
            
        elif parsed_args.command == "full":
            if not parsed_args.skip_ingest:
                logger.info("Running data ingestion pipeline")
                run_data_ingestion()
            
            if not parsed_args.skip_process:
                logger.info("Running data processing pipeline")
                run_data_processing()
            
            logger.info("Starting API server")
            start_api_server()
            
    except Exception as e:
        logger.error(f"Error running command {parsed_args.command}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 