import streamlit as st
import os
import sqlite3
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import faiss
from pathlib import Path
from llama_index.core import (
    SimpleDirectoryReader,
    load_index_from_storage,
    VectorStoreIndex,
    StorageContext,
)
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core import SQLDatabase
from sqlalchemy import create_engine
from llama_index.core.query_engine import NLSQLTableQueryEngine
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.tools import QueryEngineTool
from functools import lru_cache

# Create necessary directories
Path("data").mkdir(exist_ok=True)
Path("docs").mkdir(exist_ok=True)
Path("storage").mkdir(exist_ok=True)

# Memory optimization settings
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

@lru_cache(maxsize=1000)
def get_embedding(text):
    """Cache embeddings to avoid recomputing"""
    return st.session_state["embed_model"].get_text_embedding(text)

def setup_gpu_memory():
    """Configure GPU memory settings"""
    if torch.cuda.is_available():
        try:
            # Enable memory efficient attention
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
            # Set max memory usage
            torch.cuda.set_per_process_memory_fraction(0.7)  # Use only 70% of GPU memory
            torch.cuda.empty_cache()
        except Exception as e:
            st.warning(f"GPU memory setup failed: {str(e)}. Falling back to CPU.")

def initialize_models():
    """Initialize models with optimized settings"""
    try:
        # Use a smaller, more efficient model
        model_name = "BAAI/bge-small-en-v1.5"
        
        embeddings = HuggingFaceEmbedding(
            model_name=model_name,
            trust_remote_code=True,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        
        # Initialize LLM with memory efficient settings
        llm = HuggingFaceLLM(
            model_name="microsoft/phi-2",  # Using a smaller model
            context_window=2048,
            max_new_tokens=256,
            device_map="auto",
            model_kwargs={
                "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
                "low_cpu_mem_usage": True,
            },
            generate_kwargs={"temperature": 0.7, "top_p": 0.95},
        )
        
        return embeddings, llm
    except Exception as e:
        st.error(f"Error initializing models: {str(e)}")
        raise

def save_uploaded_file(uploaded_file, directory):
    try:
        file_path = os.path.join(directory, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return file_path
    except Exception as e:
        st.error(f"Error saving uploaded file: {str(e)}")
        return None

def preprocess_csv(csv_path):
    try:
        data = pd.read_csv(csv_path)
        required_columns = ["ACCOUNTDOCID", "BLENDED_RISK_SCORE", "AI_RISK_SCORE", 
                          "STAT_SCORE", "RULES_RISK_SCORE", "CONTROL_DEVIATION", 
                          "MONITORING_DEVIATION"]
        
        # Check if all required columns exist
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
            
        data = data[required_columns]
        data['CONTROL_DEVIATION'] = data['CONTROL_DEVIATION'].fillna("No reason given")
        data['MONITORING_DEVIATION'] = data['MONITORING_DEVIATION'].fillna("No reason given")
        
        csv_file_path = 'data/transaction_scores_processed.csv'
        data.to_csv(csv_file_path, index=False)
        return csv_file_path
    except Exception as e:
        st.error(f"Error preprocessing CSV: {str(e)}")
        return None

def process_pdf(pdf_path):
    try:
        faiss_index = faiss.IndexFlatIP(1536)
        documents = SimpleDirectoryReader("docs").load_data()
        
        if not documents:
            st.warning("No documents found in the docs directory.")
            return None

        vector_store = FaissVectorStore(faiss_index=faiss_index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(
            documents, storage_context=storage_context
        )
        index.storage_context.persist()
        return index
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return None

def build_sql_index():
    try:
        if not os.path.exists("data/transaction_scores_processed.csv"):
            st.error("Processed CSV file not found. Please upload and process a CSV file first.")
            return None

        transaction_data = pd.read_csv("data/transaction_scores_processed.csv")
        conn = sqlite3.connect("SCORES.db")
        transaction_data.to_sql("transaction_score", conn, if_exists="replace", index=False)
        conn.close()

        engine = create_engine("sqlite:///SCORES.db")
        sql_db = SQLDatabase(engine, include_tables=["transaction_score"])
        sql_query_engine = NLSQLTableQueryEngine(
            sql_database=sql_db,
            tables=["transaction_score"],
        )
        return sql_query_engine
    except Exception as e:
        st.error(f"Error building SQL index: {str(e)}")
        return None

def build_vector_index():
    try:
        if not os.path.exists("./storage"):
            st.error("Storage directory not found. Please process a PDF file first.")
            return None

        vector_store = FaissVectorStore.from_persist_dir("./storage")
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store,
            persist_dir="./storage"
        )
        
        # Use memory-mapped storage for large indices
        index_path = "index.faiss"
        faiss.write_index(vector_store.index, index_path)
        vector_store.index = faiss.read_index(index_path)
        
        index = load_index_from_storage(storage_context=storage_context)
        return index
    except Exception as e:
        st.error(f"Error building vector index: {str(e)}")
        return None

def get_sql_tool(sql_query_engine):
    from llama_index.core.tools import QueryEngineTool

    sql_tool = QueryEngineTool.from_defaults(
        query_engine=sql_query_engine,
        description=(
            "Useful for translating a natural language query into a SQL query over"
            " 2 tables namely transaction_score and accountdoc_score."
            " The tables contain the risk scores of transactions and account documents."
            " There are blended, ai, stat and rules risk scores along with conrol deviation " 
            "and monitoring deviation reasons for each transaction."
        ),
    )
    
    return sql_tool
    
    
def get_vector_tool(index):  
    from llama_index.core.retrievers import VectorIndexAutoRetriever
    from llama_index.core.vector_stores import MetadataInfo, VectorStoreInfo
    from llama_index.core.query_engine import RetrieverQueryEngine


    vector_store_info = VectorStoreInfo(
        content_info="Info about the Risk framework used for this project which gives the executive summary of the risk framework for anomaly detection. It describes the rules framework along with the list of accounting rules related to the business. Containes the description of the stat framework that uses an isolation forest model for detecting anomalies. The AI framework is made up of an autoencoder architecture. ", 
        metadata_info=[
            MetadataInfo(
                name="title", type="str", description="Rules and Architecture of Risk framework"
            ),
        ],
    )
    vector_auto_retriever = VectorIndexAutoRetriever(
        index, vector_store_info=vector_store_info
    )

    retriever_query_engine = RetrieverQueryEngine.from_args(
        vector_auto_retriever,
    )
    vector_tool = QueryEngineTool.from_defaults(
    query_engine=retriever_query_engine,
    description=(
        f"Useful for answering semantic questions related to the Risk framework which containes information avout the AI model, Stat model and Rules framework"
    ),
    )
    return vector_tool

def create_query_engine(sql_tool, vector_tool):
    from llama_index.core.query_engine import SQLAutoVectorQueryEngine

    query_engine = SQLAutoVectorQueryEngine(
        sql_tool, vector_tool)
    return query_engine
        
def main():
    try:
        # Initialize GPU memory settings
        setup_gpu_memory()
        
        if "embed_model" not in st.session_state or "llm" not in st.session_state:
            embeddings, llm = initialize_models()
            st.session_state["embed_model"] = embeddings
            st.session_state["llm"] = llm
        
        Settings.embed_model = st.session_state["embed_model"]
        Settings.llm = st.session_state["llm"]
        
        st.title("LLM Chat Interface with File Uploads")
        
        # Add memory usage indicator
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024**3
            st.sidebar.info(f"GPU Memory Usage: {gpu_memory:.2f} GB")
        
        # File uploads
        csv_files = st.file_uploader("Upload CSV files", type=["csv"], accept_multiple_files=True)
        pdf_file = st.file_uploader("Upload a PDF file", type="pdf")

        # Process and store files
        if csv_files:
            st.info("Processing CSV files...")
            if not os.path.exists("data"):
                os.makedirs("data")
            for csv_file in csv_files:
                file_path = save_uploaded_file(csv_file, "data")
                if file_path:
                    if preprocess_csv(file_path):
                        st.success(f"Successfully processed {csv_file.name}")
                    
        if pdf_file:
            st.info("Processing PDF file...")
            if not os.path.exists("docs"):
                os.makedirs("docs")
            pdf_path = save_uploaded_file(pdf_file, "docs")
            if pdf_path and process_pdf(pdf_path):
                st.success(f"Successfully processed {pdf_file.name}")

        # Button to create database and vector store
        if st.button("Create Database and Vector Store"):
            with st.spinner("Creating indexes..."):
                sql_query_engine = build_sql_index()
                vector_index = build_vector_index()
                
                if sql_query_engine and vector_index:
                    sql_tool = get_sql_tool(sql_query_engine)
                    vector_tool = get_vector_tool(vector_index)
                    query_engine = create_query_engine(sql_tool, vector_tool)
                    st.session_state["query_engine"] = query_engine
                    st.success("Indexes created successfully!")
                else:
                    st.error("Failed to create indexes. Please check the error messages above.")

        if "query_engine" in st.session_state:
            st.write("### Chat with LLM")
            user_input = st.text_input("You:", "")
            
            if "messages" not in st.session_state:
                st.session_state["messages"] = []
            
            if user_input:
                try:
                    # Clear GPU cache before processing
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    # Process query with memory management
                    with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                        response = st.session_state["query_engine"].query(user_input)
                    
                    st.session_state["messages"].append({"role": "user", "content": user_input})
                    st.session_state["messages"].append({"role": "assistant", "content": str(response)})
                    
                    # Display chat history
                    for message in st.session_state["messages"]:
                        st.write(f"**{message['role'].title()}:** {message['content']}")
                        
                except Exception as e:
                    st.error(f"Error processing query: {str(e)}")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
    except Exception as e:
        st.error(f"Application error: {str(e)}")

if __name__ == "__main__":
    main()
