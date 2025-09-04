# Prerequisites

1. Python Environment: Python 3.8+
2. OpenAI API Key: Regular OpenAI API key (not Azure)
3. Dependencies: Install from requirements.txt

## Step-by-Step Setup

1. Installation

cd multimodal_rag_for_industry
pip install -r requirements.txt

2. Configuration Changes for Regular OpenAI

A. Modify src/utils/azure_config.py

Replace the Azure configuration with OpenAI configuration:

def get_openai_config():
    return {
        'gpt4': {
            'model': 'gpt-4',
            'api_type': 'openai',
        },
        'gpt4_vision': {
            'model': 'gpt-4-vision-preview',
            'api_type': 'openai',
        },
        'text_embedding_3': {
            'model': 'text-embedding-3-small',
            'api_type': 'openai',
        },
        'gpt3.5': {
            'model': 'gpt-3.5-turbo',
            'api_type': 'openai',
        },
    }

B. Environment Variables

Instead of Azure environment variables, set:
export OPENAI_API_KEY="your-openai-api-key-here"
export GPT4V_API_KEY="your-openai-api-key-here"
export GPT4V_ENDPOINT="https://api.openai.com/v1/chat/completions"

C. Update Model Loading Files

You'll need to modify several files to use regular OpenAI instead of Azure OpenAI:

Files to modify:
- src/question_answering/rag/separate_vector_stores/dual_retrieval.py:39-42
- src/question_answering/rag/separate_vector_stores/dual_retrieval.py:176-179
- src/question_answering/rag/single_vector_store/retrieval.py:37-40
- All RAG pipeline files that use AzureOpenAIEmbeddings

Replace AzureOpenAIEmbeddings with OpenAIEmbeddings and update parameters:
from langchain_openai import OpenAIEmbeddings

# Replace this:
# self.embeddings = AzureOpenAIEmbeddings(model=config["model_version"],
#                                        azure_endpoint=config["openai_endpoint"],
#                                        api_version=config["openai_api_version"],
#                                        openai_api_key=os.getenv("AZURE_OPENAI_API_KEY_EMBEDDING"))

# With this:
self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small",
                                openai_api_key=os.getenv("OPENAI_API_KEY"))

3. Prepare Your Dataset

A. Directory Structure

Create the following structure for your data:
your_data/
├── pdfs/                    # Your PDF documents
├── reference_qa.xlsx        # Your Q&A pairs
├── extracted_data/          # Will be created automatically
├── images/                  # Will be created automatically
└── vector_stores/           # Will be created automatically

B. Q&A File Format

Create an Excel file (reference_qa.xlsx) with these columns:
- question: Your questions
- reference_answer: Expected answers
- doc_id: Source document identifier
- page_number: Page number in source document
- company: Company/source identifier (optional)

Example:
question,reference_answer,doc_id,page_number,company
"What is the operating voltage?","The operating voltage is 230V AC","manual_v1.pdf",15,"YourCompany"
"How to perform maintenance?","Follow the 5-step maintenance procedure","manual_v1.pdf",45,"YourCompany"

4. Update Configuration Paths

Modify src/rag_env.py:

# Your paths
REFERENCE_QA = "path/to/your/reference_qa.xlsx"
MANUALS_DIR = "path/to/your/pdfs/"
INPUT_DATA = "path/to/your/extracted_texts_and_imgs.parquet"
IMAGES_DIR = "path/to/your/images/"

# Output directories
IMG_SUMMARIES_CACHE_DIR = "path/to/your/image_summaries/"
TEXT_SUMMARIES_CACHE_DIR = "path/to/your/text_summaries/"
VECTORSTORE_PATH_CLIP_SINGLE = "path/to/your/vector_stores/clip/"
VECTORSTORE_PATH_SUMMARIES_SINGLE = "path/to/your/vector_stores/summaries/"
RAG_OUTPUT_DIR = "path/to/your/rag_outputs/"
EVAL_RESULTS_PATH = "path/to/your/evaluation_results/"

5. Data Processing Pipeline

Step 1: Extract Text and Images from PDFs

python src/data_extraction/pdf_context_extractor.py
This creates a parquet file with extracted texts and images.

Step 2: Generate Image Summaries (Optional but Recommended)

python src/data_summarization/context_summarization.py
This creates textual summaries of images using GPT-4 Vision.

Step 3: Build Vector Stores

The vector stores are built automatically when you run the RAG experiments.

6. Run Experiments

Text-only RAG:

python src/question_answering/rag/run_text_only_rag.py

Image-only RAG:

python src/question_answering/rag/run_image_only_rag.py

Multimodal RAG:

python src/question_answering/rag/run_multimodal_rag.py

7. Evaluate Results

python src/evaluation/evaluate_rag_pipeline.py

Key Configuration Options

Model Selection (rag_env.py):

- MODEL_TYPE = 'gpt4_vision': Use GPT-4 Vision for answer generation and image summarization
- EMBEDDING_MODEL_TYPE = "openai": Use OpenAI's text-embedding-3-small

Multimodal Approaches:

1. CLIP Embeddings: Images embedded using CLIP, stored in separate vector stores
2. Image Summaries: Images converted to text summaries, stored in combined vector store

Cost Considerations with OpenAI

- GPT-4 Vision: More expensive, better quality
- GPT-3.5: Cheaper alternative for text-only tasks
- Embeddings: text-embedding-3-small is cost-effective
- Consider batch processing for large datasets

Troubleshooting

1. API Rate Limits: Add delays between API calls if hitting rate limits
2. Memory Issues: Process documents in smaller batches
3. PDF Processing: Ensure PDFs are text-searchable; scanned PDFs may need OCR
4. Image Quality: Higher resolution images work better with GPT-4 Vision

Expected Output

- RAG Outputs: JSON files with questions, generated answers, and contexts used
- Evaluation Results: Metrics including answer correctness, relevancy, and faithfulness
- Vector Stores: Persistent storage of embeddings for future use

This setup allows you to run the complete multimodal RAG pipeline with your own documents and regular OpenAI APIs instead of Azure
OpenAI.
