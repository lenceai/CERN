"""
Fine-tuning pipeline for creating specialized CERN Magazine models
"""

import os
import logging
import argparse
import pandas as pd

from cern_mag_llmops.model.fine_tuning import FineTuningManager
from cern_mag_llmops.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(settings.DATA_DIR, 'fine_tuning_pipeline.log'))
    ]
)
logger = logging.getLogger(__name__)


def run_fine_tuning(chunks_path=None, output_dir=None, num_qa_pairs=100, base_model="gpt-3.5-turbo"):
    """
    Run the fine-tuning pipeline
    
    Args:
        chunks_path: Path to the CSV file with processed chunks
        output_dir: Directory to save fine-tuning files and results
        num_qa_pairs: Number of QA pairs to generate
        base_model: Base model to fine-tune
        
    Returns:
        dict: Pipeline results
    """
    logger.info("Starting fine-tuning pipeline")
    
    # Set default chunks path if not provided
    if not chunks_path:
        chunks_path = os.path.join(settings.DATA_DIR, "processed_text", "processed_chunks.csv")
    
    # Check if chunks file exists
    if not os.path.exists(chunks_path):
        logger.error(f"Chunks file not found: {chunks_path}")
        raise FileNotFoundError(f"Chunks file not found: {chunks_path}")
    
    # Load chunks
    logger.info(f"Loading chunks from {chunks_path}")
    chunks_df = pd.read_csv(chunks_path)
    logger.info(f"Loaded {len(chunks_df)} chunks")
    
    # Initialize fine-tuning manager
    manager = FineTuningManager(output_dir=output_dir)
    
    # Run fine-tuning pipeline
    logger.info(f"Running fine-tuning pipeline with {num_qa_pairs} QA pairs on {base_model}")
    result = manager.run_full_fine_tuning_pipeline(
        chunks_df=chunks_df,
        num_qa_pairs=num_qa_pairs,
        base_model=base_model
    )
    
    # Log results
    if result["status"] == "success":
        logger.info("Fine-tuning pipeline completed successfully")
        logger.info(f"Fine-tuned model ID: {result['model_id']}")
        logger.info(f"Training file: {result['training_file']}")
        
        # Set environment variable for the fine-tuned model
        os.environ["FINE_TUNED_MODEL"] = result["model_id"]
        logger.info(f"Set FINE_TUNED_MODEL environment variable to {result['model_id']}")
        
        # Suggest adding to .env file
        logger.info("To use this model in the future, add the following to your .env file:")
        logger.info(f"FINE_TUNED_MODEL={result['model_id']}")
    else:
        logger.error(f"Fine-tuning pipeline failed: {result.get('error', 'Unknown error')}")
    
    return result


def main():
    """Main function to run the fine-tuning pipeline from command line"""
    parser = argparse.ArgumentParser(description="CERN Magazine fine-tuning pipeline")
    parser.add_argument("--chunks-path", type=str, 
                        default=os.path.join(settings.DATA_DIR, "processed_text", "processed_chunks.csv"),
                        help="Path to the CSV file with processed chunks")
    parser.add_argument("--output-dir", type=str, 
                        default=os.path.join(settings.MODELS_DIR, "fine_tuning"),
                        help="Directory to save fine-tuning files and results")
    parser.add_argument("--num-qa-pairs", type=int, default=100,
                        help="Number of QA pairs to generate")
    parser.add_argument("--base-model", type=str, default="gpt-3.5-turbo",
                        help="Base model to fine-tune")
    
    args = parser.parse_args()
    
    run_fine_tuning(
        chunks_path=args.chunks_path,
        output_dir=args.output_dir,
        num_qa_pairs=args.num_qa_pairs,
        base_model=args.base_model
    )


if __name__ == "__main__":
    main() 