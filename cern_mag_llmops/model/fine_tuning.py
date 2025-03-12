"""
Fine-tuning components for creating specialized CERN Magazine models
"""

import os
import json
import logging
import time
import pandas as pd
from openai import OpenAI
from typing import List, Dict, Any, Optional, Tuple

from cern_mag_llmops.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(settings.DATA_DIR, 'fine_tuning.log'))
    ]
)
logger = logging.getLogger(__name__)


class FineTuningManager:
    """
    Manager for fine-tuning OpenAI models on CERN Magazine data
    """
    
    def __init__(self, output_dir=None):
        """
        Initialize the fine-tuning manager
        
        Args:
            output_dir: Directory to save fine-tuning files and results
        """
        self.output_dir = output_dir or os.path.join(settings.MODELS_DIR, "fine_tuning")
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize OpenAI client
        self.client = OpenAI()
    
    def generate_qa_pairs(self, chunks_df: pd.DataFrame, num_pairs: int = 100) -> List[Dict[str, str]]:
        """
        Generate question-answer pairs from document chunks for fine-tuning
        
        Args:
            chunks_df: DataFrame with document chunks
            num_pairs: Number of QA pairs to generate
            
        Returns:
            list: List of QA pair dictionaries
        """
        logger.info(f"Generating {num_pairs} QA pairs from {len(chunks_df)} chunks")
        
        qa_pairs = []
        samples = chunks_df.sample(min(num_pairs, len(chunks_df)))
        
        for i, row in enumerate(samples.iterrows()):
            idx, chunk = row
            
            try:
                if pd.isna(chunk['text']) or len(chunk['text'].strip()) < 100:
                    continue
                
                # Generate a question from the chunk content
                prompt = f"""Generate a specific, detailed question about CERN research that can be answered using only the following text. The question should be focused on physics concepts, experimental results, or technical details mentioned in the text:

                {chunk['text'][:1000]}
                
                Question:"""
                
                response = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=50,
                    temperature=0.7
                )
                
                question = response.choices[0].message.content.strip()
                
                # Generate a detailed answer based on the chunk
                answer_prompt = f"""Based only on the following content from a CERN Courier article, provide a detailed, accurate answer to the question. Only include information that is present in the provided text:

                Content: {chunk['text']}
                
                Question: {question}
                
                Answer:"""
                
                answer_response = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": answer_prompt}],
                    max_tokens=500,
                    temperature=0.3
                )
                
                answer = answer_response.choices[0].message.content.strip()
                
                # Add to QA pairs
                qa_pairs.append({
                    "question": question,
                    "answer": answer,
                    "source": chunk['filename'] if 'filename' in chunk else "Unknown",
                    "metadata": {
                        "year": chunk['year'] if 'year' in chunk else "Unknown",
                        "issue": chunk['issue'] if 'issue' in chunk else "Unknown"
                    }
                })
                
                logger.info(f"Generated QA pair {i+1}/{num_pairs}")
                
                # Sleep to avoid rate limits
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error generating QA pair for chunk {idx}: {e}")
                continue
        
        logger.info(f"Successfully generated {len(qa_pairs)} QA pairs")
        return qa_pairs
    
    def prepare_training_data(self, qa_pairs: List[Dict[str, str]]) -> str:
        """
        Prepare training data in JSONL format for fine-tuning
        
        Args:
            qa_pairs: List of QA pair dictionaries
            
        Returns:
            str: Path to the training data file
        """
        logger.info(f"Preparing training data from {len(qa_pairs)} QA pairs")
        
        training_data = []
        
        for pair in qa_pairs:
            # Create the training example in Chat format
            training_example = {
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant specializing in CERN research, particle physics, and high-energy physics. Provide detailed, accurate information based on CERN Courier articles."},
                    {"role": "user", "content": pair["question"]},
                    {"role": "assistant", "content": pair["answer"]}
                ]
            }
            training_data.append(training_example)
        
        # Save to JSONL file
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        output_file = os.path.join(self.output_dir, f"cern_training_data_{timestamp}.jsonl")
        
        with open(output_file, 'w') as f:
            for example in training_data:
                f.write(json.dumps(example) + '\n')
        
        logger.info(f"Training data saved to {output_file}")
        return output_file
    
    def upload_training_file(self, file_path: str) -> str:
        """
        Upload training data file to OpenAI
        
        Args:
            file_path: Path to the training data file
            
        Returns:
            str: OpenAI file ID
        """
        logger.info(f"Uploading training file {file_path} to OpenAI")
        
        try:
            with open(file_path, 'rb') as f:
                response = self.client.files.create(
                    file=f,
                    purpose="fine-tune"
                )
            
            file_id = response.id
            logger.info(f"File uploaded successfully with ID: {file_id}")
            return file_id
            
        except Exception as e:
            logger.error(f"Error uploading training file: {e}")
            raise
    
    def start_fine_tuning_job(self, training_file_id: str, model: str = "gpt-3.5-turbo") -> str:
        """
        Start a fine-tuning job
        
        Args:
            training_file_id: OpenAI file ID for training data
            model: Base model to fine-tune
            
        Returns:
            str: Fine-tuning job ID
        """
        logger.info(f"Starting fine-tuning job with file {training_file_id} on model {model}")
        
        try:
            job = self.client.fine_tuning.jobs.create(
                training_file=training_file_id,
                model=model,
                hyperparameters={
                    "n_epochs": 3,
                }
            )
            
            job_id = job.id
            logger.info(f"Fine-tuning job created with ID: {job_id}")
            return job_id
            
        except Exception as e:
            logger.error(f"Error starting fine-tuning job: {e}")
            raise
    
    def monitor_fine_tuning_job(self, job_id: str) -> Dict[str, Any]:
        """
        Monitor a fine-tuning job
        
        Args:
            job_id: Fine-tuning job ID
            
        Returns:
            dict: Job information
        """
        logger.info(f"Monitoring fine-tuning job {job_id}")
        
        while True:
            try:
                job = self.client.fine_tuning.jobs.retrieve(job_id)
                status = job.status
                
                logger.info(f"Job status: {status}")
                
                if status == "succeeded":
                    logger.info(f"Fine-tuning succeeded! Model ID: {job.fine_tuned_model}")
                    return {
                        "status": "succeeded",
                        "model_id": job.fine_tuned_model,
                        "training_tokens": getattr(job, "trained_tokens", 0),
                        "job": job
                    }
                elif status in ["failed", "cancelled"]:
                    logger.error(f"Fine-tuning {status}: {getattr(job, 'error', 'Unknown error')}")
                    return {"status": status, "error": getattr(job, "error", "Unknown error"), "job": job}
                
                # Wait before checking again
                time.sleep(60)
                
            except Exception as e:
                logger.error(f"Error monitoring fine-tuning job: {e}")
                time.sleep(60)
    
    def evaluate_fine_tuned_model(self, model_id: str, test_questions: List[str]) -> Dict[str, Any]:
        """
        Evaluate a fine-tuned model
        
        Args:
            model_id: Fine-tuned model ID
            test_questions: List of test questions
            
        Returns:
            dict: Evaluation results
        """
        logger.info(f"Evaluating fine-tuned model {model_id}")
        
        results = []
        
        for question in test_questions:
            try:
                start_time = time.time()
                
                response = self.client.chat.completions.create(
                    model=model_id,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant specializing in CERN research, particle physics, and high-energy physics."},
                        {"role": "user", "content": question}
                    ],
                    temperature=0
                )
                
                answer = response.choices[0].message.content
                end_time = time.time()
                
                results.append({
                    "question": question,
                    "answer": answer,
                    "response_time": round(end_time - start_time, 2)
                })
                
                logger.info(f"Evaluated question: {question[:50]}...")
                
                # Sleep to avoid rate limits
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error evaluating question: {e}")
                results.append({
                    "question": question,
                    "answer": f"Error: {str(e)}",
                    "response_time": 0,
                    "error": str(e)
                })
        
        # Save evaluation results
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        eval_file = os.path.join(self.output_dir, f"evaluation_results_{timestamp}.json")
        
        with open(eval_file, 'w') as f:
            json.dump({"model_id": model_id, "results": results}, f, indent=2)
        
        logger.info(f"Evaluation results saved to {eval_file}")
        
        return {
            "model_id": model_id,
            "results": results,
            "output_file": eval_file
        }
    
    def run_full_fine_tuning_pipeline(
        self, 
        chunks_df: pd.DataFrame, 
        num_qa_pairs: int = 100,
        base_model: str = "gpt-3.5-turbo"
    ) -> Dict[str, Any]:
        """
        Run the complete fine-tuning pipeline
        
        Args:
            chunks_df: DataFrame with document chunks
            num_qa_pairs: Number of QA pairs to generate
            base_model: Base model to fine-tune
            
        Returns:
            dict: Pipeline results
        """
        try:
            # Generate QA pairs
            qa_pairs = self.generate_qa_pairs(chunks_df, num_qa_pairs)
            
            # Prepare training data
            training_file = self.prepare_training_data(qa_pairs)
            
            # Upload training file
            file_id = self.upload_training_file(training_file)
            
            # Start fine-tuning job
            job_id = self.start_fine_tuning_job(file_id, base_model)
            
            # Monitor job
            job_result = self.monitor_fine_tuning_job(job_id)
            
            if job_result["status"] == "succeeded":
                model_id = job_result["model_id"]
                
                # Extract test questions from QA pairs
                test_questions = [pair["question"] for pair in qa_pairs[-10:]]
                
                # Evaluate model
                eval_result = self.evaluate_fine_tuned_model(model_id, test_questions)
                
                return {
                    "status": "success",
                    "model_id": model_id,
                    "job_id": job_id,
                    "training_file": training_file,
                    "evaluation": eval_result
                }
            else:
                return {
                    "status": "failed",
                    "error": job_result.get("error", "Unknown error"),
                    "job_id": job_id,
                    "training_file": training_file
                }
                
        except Exception as e:
            logger.error(f"Error in fine-tuning pipeline: {e}")
            return {"status": "error", "error": str(e)}


def main():
    """Test the fine-tuning pipeline with sample data"""
    try:
        # Check if processed chunks exist
        chunks_path = os.path.join(settings.DATA_DIR, "processed_text", "processed_chunks.csv")
        
        if not os.path.exists(chunks_path):
            logger.error(f"Processed chunks not found at {chunks_path}")
            print(f"Error: Processed chunks not found at {chunks_path}")
            print("Please run the data processing pipeline first.")
            return
        
        # Load chunks
        chunks_df = pd.read_csv(chunks_path)
        
        # Create manager and run pipeline with a small number of examples for testing
        manager = FineTuningManager()
        
        # Generate just a few QA pairs for testing
        qa_pairs = manager.generate_qa_pairs(chunks_df, num_pairs=5)
        
        # Prepare training data
        training_file = manager.prepare_training_data(qa_pairs)
        
        print(f"Generated {len(qa_pairs)} QA pairs for testing")
        print(f"Training data saved to {training_file}")
        print("\nTo run the full fine-tuning pipeline, use the pipeline module.")
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        print(f"Error: {e}")


if __name__ == "__main__":
    main() 