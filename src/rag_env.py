"""
Configuration for the RAG pipeline. Replace the paths with your own actual paths.
The data used here is small sample data to show the expected format, not the real data.
This toy data will not lead to good results.
"""

# model to use for answer synthesis and image summarization. ('gpt4_vision', otherwise LLaVA will be used)
MODEL_TYPE  = 'gpt4_vision'
# text embedding model, set to 'openai' to use text-embeding-3-small, otherwise bge-m3 will be used
EMBEDDING_MODEL_TYPE = "openai"

DATASET_ID = 0

DATASET_NAME = [
    "DocBench",
    "LongDocURL",
    "MMLongBenchDoc",
    "OHRBench",
    "SlideVQA",
][DATASET_ID]

# excel file containing questions and reference answers
REFERENCE_QA = f"/home/nano/projects/multimodal_rag_for_industry/data/{DATASET_NAME}/reference_qa.xlsx"

# directory containing the pdf files from which to extract texts and images
MANUALS_DIR = f"/home/nano/projects/multimodal_rag_for_industry/data/{DATASET_NAME}/pdfs/"

MANUALS_DIR = [
    "./home/nano/projects/ChunkQuest/data/datasets/DocBench",
    "./home/nano/projects/ChunkQuest/data/datasets/LongDocURL/documents",
    "./home/nano/projects/ChunkQuest/data/datasets/MMLongBenchDoc/documents",
    "./home/nano/projects/ChunkQuest/data/datasets/OHRBench/documents",
    "./home/nano/projects/ChunkQuest/data/datasets/SlideVQA/test/documents",
][DATASET_ID]


# parquet file where extracted texts and image bytes are stored
INPUT_DATA = f'/home/nano/projects/multimodal_rag_for_industry/data/{DATASET_NAME}/extracted_texts_and_imgs.parquet'

# directory where extracted images are stored
IMAGES_DIR = f"/home/nano/projects/multimodal_rag_for_industry/data/{DATASET_NAME}/images"

# directories containing csv files with text summaries or image summaries
IMG_SUMMARIES_CACHE_DIR = f"/home/nano/projects/multimodal_rag_for_industry/data/{DATASET_NAME}/image_summaries_test"
TEXT_SUMMARIES_CACHE_DIR = f"/home/nano/projects/multimodal_rag_for_industry/data/{DATASET_NAME}/text_summaries"

# directories where vector stores are saved
VECTORSTORE_PATH_CLIP_SINGLE = f"/home/nano/projects/multimodal_rag_for_industry/data/{DATASET_NAME}/vec_and_doc_stores/clip"
VECTORSTORE_PATH_CLIP_SEPARATE = f"/home/nano/projects/multimodal_rag_for_industry/data/{DATASET_NAME}/vec_and_doc_stores/clip_dual"
VECTORSTORE_PATH_SUMMARIES_SINGLE = f"/home/nano/projects/multimodal_rag_for_industry/data/{DATASET_NAME}/vec_and_doc_stores/image_summaries"
VECTORSTORE_PATH_SUMMARIES_SEPARATE = f"/home/nano/projects/multimodal_rag_for_industry/data/{DATASET_NAME}/vec_and_doc_stores/image_summaries_dual"
VECTORSTORE_PATH_IMAGE_ONLY = f"./home/nano/projects/multimodal_rag_for_industry/data/{DATASET_NAME}/vec_and_doc_stores/image_only"
VECTORSTORE_PATH_TEXT_ONLY = f"./home/nano/projects/multimodal_rag_for_industry/data/{DATASET_NAME}/vec_and_doc_stores/text_only"

# directory where the output of a RAG pipeline is stored
RAG_OUTPUT_DIR = f"./home/nano/projects/multimodal_rag_for_industry/data/{DATASET_NAME}/rag_outputs"

# directory where the evaluation results for a RAG pipeline are stored
EVAL_RESULTS_PATH = f"/home/nano/projects/multimodal_rag_for_industry/data/{DATASET_NAME}/rag_evaluation_results"
