"""
Comparison between RAG and fine-tuned models
"""

import os
import logging
import json
import time
from typing import List, Dict, Any, Optional
import pandas as pd
import matplotlib.pyplot as plt
from openai import OpenAI

from cern_mag_llmops.config import settings
from cern_mag_llmops.model.rag_model import RAGModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(settings.DATA_DIR, 'model_comparison.log'))
    ]
)
logger = logging.getLogger(__name__)


class ModelComparison:
    """
    Compare performance between RAG and fine-tuned models
    """
    
    def __init__(self, fine_tuned_model_id=None):
        """
        Initialize the model comparison
        
        Args:
            fine_tuned_model_id: ID of the fine-tuned model to compare
        """
        self.fine_tuned_model_id = fine_tuned_model_id or settings.FINE_TUNED_MODEL
        if not self.fine_tuned_model_id:
            logger.warning("No fine-tuned model ID provided. Only RAG model will be used.")
        
        # Initialize OpenAI client for fine-tuned model
        self.client = OpenAI()
        
        # Initialize RAG model
        self.rag_model = RAGModel()
        
        # Create output directory
        self.output_dir = os.path.join(settings.MODELS_DIR, "comparison")
        os.makedirs(self.output_dir, exist_ok=True)
    
    def query_fine_tuned_model(self, question: str) -> Dict[str, Any]:
        """
        Query the fine-tuned model
        
        Args:
            question: Question to ask
            
        Returns:
            dict: Response data
        """
        if not self.fine_tuned_model_id:
            return {
                "answer": "No fine-tuned model available",
                "response_time": 0,
                "error": "No fine-tuned model ID provided"
            }
        
        try:
            start_time = time.time()
            
            response = self.client.chat.completions.create(
                model=self.fine_tuned_model_id,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant specializing in CERN research, particle physics, and high-energy physics."},
                    {"role": "user", "content": question}
                ],
                temperature=0
            )
            
            answer = response.choices[0].message.content
            end_time = time.time()
            response_time = round(end_time - start_time, 2)
            
            return {
                "answer": answer,
                "response_time": response_time,
                "model": self.fine_tuned_model_id
            }
            
        except Exception as e:
            logger.error(f"Error querying fine-tuned model: {e}")
            return {
                "answer": f"Error: {str(e)}",
                "response_time": 0,
                "error": str(e),
                "model": self.fine_tuned_model_id
            }
    
    def compare_models(self, question: str) -> Dict[str, Any]:
        """
        Compare RAG and fine-tuned models for a single question
        
        Args:
            question: Question to compare
            
        Returns:
            dict: Comparison results
        """
        logger.info(f"Comparing models for question: {question}")
        
        # Query RAG model
        rag_result = self.rag_model.answer_question(question)
        
        # Query fine-tuned model if available
        if self.fine_tuned_model_id:
            ft_result = self.query_fine_tuned_model(question)
        else:
            ft_result = {
                "answer": "No fine-tuned model available",
                "response_time": 0,
                "error": "No fine-tuned model ID provided"
            }
        
        result = {
            "question": question,
            "fine_tuned": {
                "answer": ft_result["answer"],
                "response_time": ft_result["response_time"],
                "model": self.fine_tuned_model_id or "N/A"
            },
            "rag": {
                "answer": rag_result["answer"],
                "response_time": rag_result["response_time"],
                "model": rag_result["model"],
                "sources": rag_result["sources"]
            },
            "timestamp": time.time()
        }
        
        return result
    
    def run_comparison_batch(self, questions: List[str]) -> Dict[str, Any]:
        """
        Run comparison on a batch of questions
        
        Args:
            questions: List of questions to compare
            
        Returns:
            dict: Batch comparison results
        """
        logger.info(f"Running comparison batch with {len(questions)} questions")
        
        results = []
        
        for i, question in enumerate(questions):
            logger.info(f"Processing question {i+1}/{len(questions)}")
            
            result = self.compare_models(question)
            results.append(result)
            
            # Sleep to avoid rate limits
            time.sleep(1)
        
        # Calculate summary statistics
        rag_times = [r["rag"]["response_time"] for r in results]
        ft_times = [r["fine_tuned"]["response_time"] for r in results if r["fine_tuned"]["response_time"] > 0]
        
        summary = {
            "questions": len(questions),
            "rag": {
                "avg_response_time": round(sum(rag_times) / len(rag_times), 2) if rag_times else 0,
                "min_response_time": round(min(rag_times), 2) if rag_times else 0,
                "max_response_time": round(max(rag_times), 2) if rag_times else 0
            },
            "fine_tuned": {
                "avg_response_time": round(sum(ft_times) / len(ft_times), 2) if ft_times else 0,
                "min_response_time": round(min(ft_times), 2) if ft_times else 0,
                "max_response_time": round(max(ft_times), 2) if ft_times else 0
            }
        }
        
        # Save results
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        output_file = os.path.join(self.output_dir, f"comparison_results_{timestamp}.json")
        
        with open(output_file, 'w') as f:
            json.dump({"summary": summary, "results": results}, f, indent=2)
        
        logger.info(f"Comparison results saved to {output_file}")
        
        return {
            "summary": summary,
            "results": results,
            "output_file": output_file
        }
    
    def visualize_comparison(self, comparison_results: Dict[str, Any]) -> str:
        """
        Create visualization of comparison results
        
        Args:
            comparison_results: Results from run_comparison_batch
            
        Returns:
            str: Path to the saved visualization file
        """
        # Extract data for plotting
        results = comparison_results["results"]
        
        # Create dataframe for easier plotting
        data = []
        for result in results:
            data.append({
                "question": result["question"][:30] + "..." if len(result["question"]) > 30 else result["question"],
                "model": "RAG",
                "response_time": result["rag"]["response_time"]
            })
            
            if result["fine_tuned"]["response_time"] > 0:
                data.append({
                    "question": result["question"][:30] + "..." if len(result["question"]) > 30 else result["question"],
                    "model": "Fine-tuned",
                    "response_time": result["fine_tuned"]["response_time"]
                })
        
        df = pd.DataFrame(data)
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        # Bar plot of response times
        ax = plt.subplot(111)
        bars = plt.bar(range(len(df)), df["response_time"], color=df["model"].map({"RAG": "blue", "Fine-tuned": "green"}))
        
        # Add labels
        plt.xticks(range(len(df)), df["question"], rotation=45, ha="right")
        plt.ylabel("Response Time (seconds)")
        plt.title("RAG vs Fine-tuned Model Response Times")
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="blue", label="RAG"),
            Patch(facecolor="green", label="Fine-tuned")
        ]
        ax.legend(handles=legend_elements, loc="upper right")
        
        plt.tight_layout()
        
        # Save figure
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        output_file = os.path.join(self.output_dir, f"comparison_plot_{timestamp}.png")
        plt.savefig(output_file)
        plt.close()
        
        logger.info(f"Comparison visualization saved to {output_file}")
        return output_file
    
    def interactive_comparison(self):
        """
        Run an interactive comparison session
        
        This method allows users to enter questions and see responses from both models
        """
        print("CERN Research Assistant Comparison")
        print("Compare Fine-tuned model vs RAG approach")
        print("Type 'quit' to exit")
        
        while True:
            print("\nYour question: ", end="")
            question = input().strip()
            
            if question.lower() in ["quit", "exit", "q"]:
                break
            
            if not question:
                continue
            
            print("\nQuerying both models...")
            result = self.compare_models(question)
            
            print("\n" + "=" * 50)
            print(f"Question: {question}")
            print("=" * 50)
            
            print("\nFine-tuned Model Response:")
            print("-" * 30)
            print(result["fine_tuned"]["answer"])
            print(f"Response time: {result['fine_tuned']['response_time']} seconds")
            
            print("\nRAG System Response:")
            print("-" * 30)
            print(result["rag"]["answer"])
            print(f"Response time: {result['rag']['response_time']} seconds")


def main():
    """Run an interactive model comparison"""
    # Check if fine-tuned model ID is provided
    fine_tuned_model = os.getenv("FINE_TUNED_MODEL", settings.FINE_TUNED_MODEL)
    
    if not fine_tuned_model:
        print("Warning: No fine-tuned model ID provided.")
        print("Only the RAG model will be used.")
        print("Set the FINE_TUNED_MODEL environment variable to compare with a fine-tuned model.")
        print()
    
    # Create and run comparison
    comparison = ModelComparison(fine_tuned_model)
    comparison.interactive_comparison()


if __name__ == "__main__":
    main() 