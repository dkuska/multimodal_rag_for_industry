# Multimodal RAG Execution Plan for Custom Datasets

This document provides a comprehensive guide for setting up and running the Multimodal RAG system on your own PDF datasets with custom question-answer pairs.

## Overview

This system implements multimodal Retrieval-Augmented Generation (RAG) that can process both text and images from PDF documents to answer questions. It supports three main approaches:
- **Text-only RAG**: Uses only textual content
- **Image-only RAG**: Uses only visual content
- **Multimodal RAG**: Combines both text and images

## Prerequisites

### 1. System Requirements
- Python 3.8 or higher
- GPU (recommended for LLaVA model, optional for GPT-4V)
- Sufficient disk space for vector stores and intermediate data
- At least 16GB RAM (32GB recommended)

### 2. Required API Keys
You'll need API keys depending on which models you choose:

#### For GPT-4V and OpenAI Embeddings:
- **Azure OpenAI API Key**: Set as environment variable
  ```bash
  export AZURE_OPENAI_API_KEY="your-api-key-here"
  export AZURE_OPENAI_API_KEY_EMBEDDING="your-embedding-api-key-here"
  ```

#### For LLaVA (Open-source alternative):
- No API keys required, but you'll need GPU resources

## Setup Instructions

### 1. Create Virtual Environment
```bash
# Create a new virtual environment
python -m venv multimodal_rag_env

# Activate the environment
# On Linux/Mac:
source multimodal_rag_env/bin/activate
# On Windows:
multimodal_rag_env\Scripts\activate
```

### 2. Install Dependencies
```bash
# Clone the repository first if you haven't
git clone <repository-url>
cd multimodal_rag_for_industry

# Install all requirements
pip install -r requirements.txt

# Note: If you encounter issues with CLIP installation, install it separately:
pip install git+https://github.com/openai/CLIP.git@a1d071733d7111c9c014f024669f959182114e33
```

### 3. Configure Azure OpenAI Endpoints
Edit `src/utils/azure_config.py` and replace `'YOUR_ENDPOINT_HERE'` with your actual Azure OpenAI endpoints:

```python
def get_azure_config():
    return {
        'gpt4': {
            'openai_endpoint': 'https://YOUR-RESOURCE-NAME.openai.azure.com/',
            'deployment_name': 'gpt-4',
            'openai_api_version': '2024-02-15-preview',
            'model_version': 'gpt-4',
        },
        'gpt4_vision': {
            'openai_endpoint': 'https://YOUR-RESOURCE-NAME.openai.azure.com/',
            'deployment_name': 'gpt-4-vision-preview',
            'openai_api_version': '2024-02-15-preview',
            'model_version': 'gpt-4-vision-preview',
        },
        'text_embedding_3': {
            'openai_endpoint': 'https://YOUR-RESOURCE-NAME.openai.azure.com/',
            'deployment_name': 'text-embedding-3-small',
            'openai_api_version': '2024-02-15-preview',
            'model_version': 'text-embedding-3-small',
        },
        # ... other configurations
    }
```

## Data Preparation

### 1. Prepare Your PDF Files
- Create a directory containing all your PDF files
- Ensure PDFs are readable and not password-protected
- Recommended: Organize PDFs by topic or category

### 2. Format Your Question-Answer Pairs

Create an Excel file (`.xlsx`) with the following columns:

| Column Name | Description | Example |
|-------------|-------------|---------|
| `doc_id` | Unique identifier for the document/question | "Q001" |
| `question` | The question to be answered | "What is the maintenance schedule for the equipment?" |
| `answer` | The reference/correct answer | "The equipment should be maintained every 3 months..." |
| `page_number` | Page number where answer can be found | 42 |
| `company_name` | (Optional) Company or document source | "ACME Corp" |

Example Excel structure:
```
doc_id | question | answer | page_number | company_name
-------|----------|--------|-------------|-------------
Q001   | What is...? | The answer is... | 10 | ACME Corp
Q002   | How does...? | It works by... | 25 | ACME Corp
```

### 3. Create JSONL File (Alternative to Excel)

If you have a JSONL file instead of Excel, convert it to Excel format:

```python
import pandas as pd
import json

# Read JSONL file
data = []
with open('your_qa_pairs.jsonl', 'r') as f:
    for line in f:
        data.append(json.loads(line))

# Convert to DataFrame and save as Excel
df = pd.DataFrame(data)
df.to_excel('reference_qa.xlsx', index=False)
```

## Configuration

### 1. Update `src/rag_env.py`

Replace the placeholder paths with your actual paths:

```python
# Model configuration
MODEL_TYPE = 'gpt4_vision'  # or 'llava' for open-source
EMBEDDING_MODEL_TYPE = "openai"  # or 'bge-m3' for open-source

# Data paths
REFERENCE_QA = "/path/to/your/reference_qa.xlsx"
MANUALS_DIR = "/path/to/your/pdf/directory"
INPUT_DATA = "/path/to/output/extracted_texts_and_imgs.parquet"
IMAGES_DIR = "/path/to/output/images"

# Cache directories
IMG_SUMMARIES_CACHE_DIR = "/path/to/cache/image_summaries"
TEXT_SUMMARIES_CACHE_DIR = "/path/to/cache/text_summaries"

# Vector store paths
VECTORSTORE_PATH_CLIP_SINGLE = "/path/to/vectorstore/clip"
VECTORSTORE_PATH_SUMMARIES_SINGLE = "/path/to/vectorstore/summaries"
# ... other paths
```

## Execution Steps

### Step 1: Extract Content from PDFs
```bash
cd src
python data_extraction/pdf_content_extractor.py
```
This will:
- Extract text chunks from PDFs
- Extract images from PDFs
- Save results to a parquet file
- Save images to the specified directory

### Step 2: Generate Image Summaries (Optional but Recommended)
```bash
python data_summarization/context_summarization.py
```
This will:
- Generate textual descriptions of images using GPT-4V or LLaVA
- Save summaries to CSV files
- Enable better image-based retrieval

### Step 3: Create Vector Stores
The vector stores are created automatically when running the RAG pipelines, but you can pre-create them:

```bash
# For text-only RAG
python question_answering/rag/run_text_only_rag.py

# For image-only RAG
python question_answering/rag/run_image_only_rag.py

# For multimodal RAG
python question_answering/rag/run_multimodal_rag.py
```

### Step 4: Run Experiments

Choose your approach and run the corresponding script:

#### Baseline (No RAG)
```bash
python question_answering/baseline/run_baseline.py
```

#### Text-Only RAG
```bash
python question_answering/rag/run_text_only_rag.py
```

#### Image-Only RAG
```bash
python question_answering/rag/run_image_only_rag.py
```

#### Multimodal RAG
```bash
python question_answering/rag/run_multimodal_rag.py
```

#### Upper Bound (Perfect Retrieval)
```bash
python question_answering/correct_context_prompting/run_multimodal_correct_context_qa.py
```

### Step 5: Evaluate Results
```bash
python evaluation/evaluate_rag_pipeline.py
```

This evaluates:
- Answer Correctness
- Answer Relevancy
- Context Relevancy (Text/Image)
- Faithfulness (Text/Image)

## Output Structure

### Generated Files
1. **Extracted Data**: `extracted_texts_and_imgs.parquet`
   - Contains text chunks and image bytes from PDFs

2. **Image Summaries**: `image_summaries/*.csv`
   - Textual descriptions of images

3. **Vector Stores**: Various directories containing embeddings
   - CLIP embeddings for images
   - Text embeddings for documents

4. **RAG Outputs**: `rag_outputs/*.json`
   - Generated answers for each question
   - Retrieved context information

5. **Evaluation Results**: `rag_evaluation_results/*.json`
   - Performance metrics
   - Detailed evaluation scores

## Troubleshooting

### Common Issues

1. **Memory Errors**
   - Reduce batch size in processing scripts
   - Process PDFs in smaller batches
   - Use smaller embedding models

2. **API Rate Limits**
   - Add delays between API calls
   - Use exponential backoff
   - Consider batch processing

3. **PDF Extraction Issues**
   - Ensure PDFs are not corrupted
   - Check for password protection
   - Try different PDF extraction libraries

4. **GPU Memory (for LLaVA)**
   - Reduce model precision (use int8)
   - Process fewer images at once
   - Use smaller model variants

### Environment Variables Summary
```bash
# Required for OpenAI models
export AZURE_OPENAI_API_KEY="your-key"
export AZURE_OPENAI_API_KEY_EMBEDDING="your-embedding-key"

# Optional for debugging
export LOG_LEVEL="INFO"
```

## Best Practices

1. **Data Quality**
   - Ensure PDFs have good OCR quality
   - Verify question-answer pairs are accurate
   - Include page numbers for better evaluation

2. **Performance Optimization**
   - Pre-compute embeddings for large datasets
   - Use caching for image summaries
   - Consider using smaller models for testing

3. **Evaluation**
   - Start with a small subset for testing
   - Compare different approaches systematically
   - Track performance metrics over time

## Next Steps

After successfully running the pipeline:

1. Analyze evaluation results in `img/results/` directory
2. Fine-tune retrieval parameters (chunk size, top-k, etc.)
3. Experiment with different embedding models
4. Consider domain-specific fine-tuning
5. Scale up to larger datasets

## Additional Resources

- [Paper](https://arxiv.org/abs/2410.21943): Beyond Text: Optimizing RAG with Multimodal Inputs
- [Azure OpenAI Documentation](https://learn.microsoft.com/en-us/azure/cognitive-services/openai/)
- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)
- [CLIP Model](https://github.com/openai/CLIP)

---

For questions or issues, please refer to the main README.md or open an issue in the repository.
