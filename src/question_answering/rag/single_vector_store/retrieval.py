import os
import uuid
import pandas as pd
from tqdm import tqdm
from langchain.retrievers import MultiVectorRetriever
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_experimental.open_clip import OpenCLIPEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain.storage import LocalFileStore
from utils.azure_config import get_azure_config
from rag_env import IMAGES_DIR
from typing import List


class SummaryStoreAndRetriever:
    """  
    A class providing a document store and a vector store to contain texts and images and their embeddings.
    The class also provides a retriever to find documents that are relevant for a query.
    Retrieval is performed using the embeddings in the vector store, but the documents contained in the
    document store are returned. This allows image retrieval via the image summaries, while still ensuring
    that the original images associated with the summaries are returned.
  
    Attributes:
        embedding_model (str): Model used to embed the texts and image summaries.
        store_path (str): Path where the vector and document stores should be saved.
        docstore (LocalFileStore): Document store containing texts and images.
        vectorstore (Chroma): Vector store containing embedded texts and image summaries.
        retriever (MultiVectorRetriever): Retriever that encommpasses both a vector store and a document store.
    """   
    def __init__(self, embedding_model, store_path=None):
        
        # embed using openai embedding model
        if embedding_model == 'openai':
            print("Using text-embedding-3-small")
            azure_embedding_config = get_azure_config()['text_embedding_3']
            self.embeddings = OpenAIEmbeddings(model=azure_embedding_config["model_version"],
                                                    openai_api_key=os.getenv("OPENAI_API_KEY"),
                                                    chunk_size=64,
                                                    show_progress_bar=True
                                                    )
        # embed with BGE embeddings
        else:
            print("Using BGE embeddings")
            model_name = "BAAI/bge-m3"
            model_kwargs = {"device": "cuda"}
            encode_kwargs = {"normalize_embeddings": True, "batch_size": 1, "show_progress_bar":True}
            self.embeddings = HuggingFaceBgeEmbeddings(
                model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
                )

        self.store_path = store_path
        vectorstore_dir = os.path.join(self.store_path, f"{os.path.basename(self.store_path)}_vectorstore_{embedding_model}")
        docstore_dir = os.path.join(self.store_path, f"{os.path.basename(self.store_path)}_docstore_{embedding_model}")
        
        # Initialize the document store
        self.docstore = LocalFileStore(docstore_dir)
        self.id_key = "doc_id"
        self.doc_ids = []

        # Initialize the vector store with the embedding function
        self.vectorstore = Chroma(
            persist_directory=vectorstore_dir,
            embedding_function=self.embeddings,
            collection_name=f"mm_rag_with_image_summaries_{embedding_model}_embeddings"
        )
        results = self.vectorstore.get(include=["embeddings", "documents", "metadatas"])
        self.is_new_vectorstore = bool(results["embeddings"].any())

        if self.is_new_vectorstore:
            print(f"Vectorstore at path {vectorstore_dir} already exists")

        else:
            print(f"Creating new vectorstore and docstore at path {self.store_path}")

        self.retriever = MultiVectorRetriever(
            vectorstore=self.vectorstore,
            docstore=self.docstore,
            id_key=self.id_key
        )
        self.retrieved_docs = []


    def add_docs(
        self,
        doc_summaries: List[str],
        doc_contents: List[str],
        doc_filenames: List[str],
        batch_size: int = 5000
    ):
        """Add documents to the vector store and document store in batches.
        
        Args:
            doc_summaries: Either text or image summaries for the vector store.
            doc_contents: The original texts or images for the document store.
            doc_filenames: File names associated with the documents.
            batch_size: Max documents to process in a single batch.
        """
        if not self.is_new_vectorstore:
            print("Adding documents...")
            doc_ids = [str(uuid.uuid4()) for _ in doc_contents]
            
            # Process documents in batches
            for i in range(0, len(doc_summaries), batch_size):
                batch_end = min(i + batch_size, len(doc_summaries))
                print(f"Batch {i//batch_size + 1}: docs {i} to {batch_end}")
                
                # Create batch of documents
                batch_summary_docs = [
                    Document(
                        page_content=s,
                        metadata={
                            self.id_key: doc_ids[j],
                            "filename": doc_filenames[j]
                        }
                    )
                    for j, s in enumerate(doc_summaries[i:batch_end], start=i)
                ]
                
                # Add documents to vector store
                self.vectorstore.add_documents(batch_summary_docs)
                
                # Add corresponding documents to doc store
                batch_doc_pairs = list(zip(
                    doc_ids[i:batch_end],
                    map(lambda x: str.encode(x), doc_contents[i:batch_end])
                ))
                self.docstore.mset(batch_doc_pairs)
            
            print("Finished adding all documents.")
        else:
            print("Documents have already been added before, skipping...")


    def retrieve(self, query: str, limit: int) -> List[Document]:
        """
        Retrieve the most relevant documents based on the query.
        """
        self.retrieved_docs = self.retriever.invoke(query, limit=limit)

        return self.retrieved_docs
    
    
    
    
class ClipRetriever:
    """  
    A class providing a vector store to contain texts and images embedded with the multimodal embedding model CLIP.
    The vector store can also be used as retriever to find documents that are relevant for a query.
  
    Attributes:
        vectorstore_dir (str): Path where the vector store should be saved.
        images_dir (str): Directory containing the images to be embedded.
        vectorstore (Chroma): Vector store containing embedded texts and images.
    """
    def __init__(self, vectorstore_dir, images_dir=IMAGES_DIR):
        self.images_dir = images_dir

        # Create chroma vectorstore
        self.vectorstore = Chroma(
            collection_name="mm_rag_clip_photos", 
            embedding_function=OpenCLIPEmbeddings(),
            persist_directory=vectorstore_dir
        )

        results = self.vectorstore.get(include=["embeddings", "documents", "metadatas"])
        self.is_new_vectorstore = bool(results["embeddings"].any())

        if self.is_new_vectorstore:
            print(f"Vectorstore at path {vectorstore_dir} already exists")

        else:
            print(f"Creating new vectorstore at path {vectorstore_dir}")

        # Make retriever
        self.retriever = self.vectorstore.as_retriever()
        self.retrieved_docs = []
        
        
        
    def add_documents(self, images_dir: str=None, texts_df: pd.DataFrame=None, batch_size: int=8):
        """
        Add images and texts to the vector store in batches.
        
        :param images_dir: Directory containing the images to be embedded.
        :param texts_df: Dataframe containing the texts to be embedded.
        :param batch_size: Size of batches for processing images and texts (default: 64).
        """
        if not self.is_new_vectorstore:
            if images_dir:
                image_uris = self.extract_image_uris(images_dir)  
                print(f"Found {len(image_uris)} images")
                
                # Process images in batches with progress bar
                for i in tqdm(range(0, len(image_uris), batch_size), desc="Adding images to vectorstore"):
                    batch_end = min(i + batch_size, len(image_uris))
                    batch_uris = image_uris[i:batch_end]
                    batch_metadatas = [{'filename': self.extract_manual_name(uri)} for uri in batch_uris]
                    self.vectorstore.add_images(uris=batch_uris, metadatas=batch_metadatas)
                
            if texts_df is not None:
                texts = texts_df["text"].to_list()
                text_metadatas = [{'filename': doc_id} for doc_id in texts_df['doc_id']]

                # Process texts in batches with progress bar
                for i in tqdm(range(0, len(texts), batch_size), desc="Adding texts to vectorstore"):
                    batch_end = min(i + batch_size, len(texts))
                    batch_texts = texts[i:batch_end]
                    batch_metadatas = text_metadatas[i:batch_end]
                    self.vectorstore.add_texts(texts=batch_texts, metadatas=batch_metadatas)
        else:
            print("Documents have already been added before, skipping...")


        
    def extract_image_uris(self, root_path: str, image_extension: str=".png") -> List[str]:
        image_uris = []
        for subdir, dirs, files in os.walk(root_path):
            for file in files:
                if file.endswith(image_extension):
                    image_uris.append(os.path.join(subdir, file))
        return sorted(image_uris)



    def extract_manual_name(self, uri: str) -> str:
        # Split the URI into parts
        parts = uri.split('/')
        # The directory name is the second to last element
        directory_name = parts[-2]
        # Append ".pdf" to the directory name
        return f"{directory_name}.pdf"
    
    
    def retrieve(self, query: str, limit: int) -> List[Document]:
        """
        Retrieve the most relevant documents based on the query.
        """
        self.retrieved_docs = self.retriever.invoke(query, limit=limit)

        return self.retrieved_docs
