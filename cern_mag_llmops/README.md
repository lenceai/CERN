# CERN Magazine LLMOps

This project implements an LLMOps framework for processing CERN Magazine PDFs, creating a vector database for search, and deploying a RAG (Retrieval Augmented Generation) system.

## Project Structure

- **config**: Central configuration for paths, API keys, and parameters
  - `settings.py`: Configuration settings for the entire framework
  
- **data_ingestion**: Components for crawling and downloading CERN Courier PDFs
  - `pdf_crawler.py`: Crawler for downloading CERN Courier PDFs
  
- **data_processing**: Utilities for extracting text from PDFs and creating embeddings
  - `pdf_processor.py`: Text extraction and preprocessing from PDFs
  - `vector_store.py`: Vector database creation for document retrieval
  
- **model**: Model management, fine-tuning, and evaluation
  - `rag_model.py`: RAG implementation for question answering
  - `fine_tuning.py`: Fine-tuning capabilities for specialized models
  - `model_comparison.py`: Comparison between RAG and fine-tuned models
  
- **api**: API endpoints for interacting with the RAG system
  - `models.py`: Pydantic models for API request/response validation
  - `server.py`: FastAPI server for interacting with the models
  
- **pipelines**: End-to-end pipelines for training and deployment
  - `data_ingestion_pipeline.py`: End-to-end data ingestion workflow
  - `data_processing_pipeline.py`: Text extraction and vector DB creation
  - `fine_tuning_pipeline.py`: Fine-tuning workflow
  
- **utils**: Shared utility functions
  - `cli.py`: Command-line interface for the framework

## Getting Started

1. Install the package:
   ```
   pip install -e .
   ```

2. Create a `.env` file with the required API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key
   ```

3. Run the data ingestion pipeline:
   ```
   cern-mag-llmops ingest
   ```

4. Process data and create the vector database:
   ```
   cern-mag-llmops process
   ```

5. Start the API server:
   ```
   cern-mag-llmops serve
   ```

## Key Features

### PDF Crawling and Processing
- Automated downloading of CERN Courier PDFs
- Text extraction and preprocessing
- Metadata extraction from filenames

### Vector Database Creation
- Text chunking with overlap
- Embedding generation using OpenAI models
- Efficient batch processing to avoid rate limits

### RAG System
- Retrieval of relevant documents for queries
- Integration with LangChain for RAG implementation
- Transparent sourcing of information

### Fine-tuning Capabilities
- Automated generation of QA pairs from document chunks
- Training data preparation in JSONL format
- Fine-tuning job management and monitoring

### Model Comparison
- Side-by-side comparison of RAG and fine-tuned models
- Performance metrics and visualization
- Interactive comparison mode

### API for Integration
- RESTful API with FastAPI
- Swagger documentation
- Health check and monitoring endpoints

### Command-line Interface
- Easy-to-use CLI for running pipelines
- Flexible configuration options
- End-to-end workflow support

## Usage

The framework can be used in several ways:

### As a Python Package
```python
from cern_mag_llmops.model.rag_model import RAGModel

rag_model = RAGModel()
result = rag_model.answer_question("What is the Higgs boson?")
```

### Via Command-line Interface
```bash
# Download PDFs
cern-mag-llmops ingest

# Process PDFs and create vector DB
cern-mag-llmops process

# Fine-tune a model
cern-mag-llmops finetune

# Start the API server
cern-mag-llmops serve

# Compare models interactively
cern-mag-llmops compare

# Run the full pipeline
cern-mag-llmops full
```

### RESTful API
The framework provides a RESTful API that can be used to interact with the models:

- `/query`: Answer questions using RAG, fine-tuned model, or both
- `/documents`: Retrieve relevant documents for a query
- `/health`: Check the health of the API

## Requirements

See `requirements.txt` for a complete list of dependencies.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 