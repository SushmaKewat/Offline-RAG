import streamlit as st
import os
import sqlite3
import pandas as pd

import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd
import faiss
import json

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

def save_uploaded_file(uploaded_file, directory):
    file_path = os.path.join(directory, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def preprocess_csv(csv_path):
    data = pd.read_csv(csv_path)
    data = data[["ACCOUNTDOCID","BLENDED_RISK_SCORE", "AI_RISK_SCORE", "STAT_SCORE", "RULES_RISK_SCORE", "CONTROL_DEVIATION", "MONITORING_DEVIATION"]]
    data['CONTROL_DEVIATION'] = data['CONTROL_DEVIATION'].fillna("No reason given")
    data['MONITORING_DEVIATION'] = data['MONITORING_DEVIATION'].fillna("No reason given")
    
    csv_file_path = 'data/transaction_scores_processed.csv'
    data.to_csv(csv_file_path, index=False)

    print(f"DataFrame has been saved to {csv_file_path}")
    return csv_file_path

def process_pdf(pdf_path):
    faiss_index = faiss.IndexFlatIP(2048)

    documents = SimpleDirectoryReader("docs").load_data()

    vector_store = FaissVectorStore(faiss_index=faiss_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context
    )
    index.storage_context.persist()


def build_sql_index():
    # Load structured data into SQLite database
    transaction_data = pd.read_csv("data/transaction_scores_processed.csv")
    # accountdoc_data = pd.read_csv("data/account_scores_processed.csv")

    conn = sqlite3.connect("SCORES.db")

    transaction_data.to_sql("transaction_score", conn, if_exists="replace", index=False)
    # accountdoc_data.to_sql("accountdoc_score", conn, if_exists="replace", index=False)

    print("Structured data loaded into SQLite database.")
    
    engine = create_engine("sqlite:///SCORES.db") 

    sql_db = SQLDatabase(engine, include_tables=["transaction_score"]) 
    sql_query_engine = NLSQLTableQueryEngine(
    sql_database=sql_db,
    tables=["transaction_score"],
    )
    
    return sql_query_engine

def build_vector_index():
    # load index from disk
    from llama_index.core import (
        SimpleDirectoryReader,
        load_index_from_storage,
        VectorStoreIndex,
        StorageContext,
    )
    from llama_index.vector_stores.faiss import FaissVectorStore

    vector_store = FaissVectorStore.from_persist_dir("./storage")
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store, persist_dir="./storage"
    )
    index = load_index_from_storage(storage_context=storage_context)
    return index


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
        f"Useful for answering semantic questions related to the Risk framework"
    ),
    )
    return vector_tool

def create_query_engine(sql_tool, vector_tool):
    from llama_index.core.query_engine import SQLAutoVectorQueryEngine

    query_engine = SQLAutoVectorQueryEngine(
        sql_tool, vector_tool)
    return query_engine
        
def main():
    Settings.embed_model = st.session_state["embed_model"]
    Settings.llm = st.session_state["llm"]
    st.title("LLM Chat Interface with File Uploads")

    # File uploads
    csv_files = st.file_uploader("Upload CSV files", type=["csv"], accept_multiple_files=True)
    pdf_file = st.file_uploader("Upload a PDF file", type="pdf")

    # Process and store files
    if csv_files:
        st.write("Processing CSV files...")
        if not os.path.exists("data"):
            os.makedirs("data")
        for csv_file in csv_files:
            try:
                csv_path = save_uploaded_file(csv_file, "data")
                preprocess_csv(csv_path)
            except Exception as e:
                st.error(f"Error processing {csv_file.name}: {e}")
    if pdf_file:
        st.write("Processing PDF file...")
        if not os.path.exists("docs"):
            os.makedirs("docs")
        pdf_path = save_uploaded_file(pdf_file, "docs")
        process_pdf(pdf_path)

    # Button to create database and vector store
    if st.button("Create Database and Vector Store"):
        sql_query_engine = build_sql_index()
        vector_index = build_vector_index()
        sql_tool = get_sql_tool(sql_query_engine)
        vector_tool = get_vector_tool(vector_index)
        query_engine = create_query_engine(sql_tool, vector_tool)
        st.session_state["query_engine"] = query_engine

    if "query_engine" in st.session_state:
        st.write("### Chat with LLM")
        user_input = st.text_input("You:", "")

        if "messages" not in st.session_state:
            st.session_state["messages"] = []

        if user_input:
            try:
                response = st.session_state["query_engine"].query(user_input)
                st.session_state["messages"].append((user_input, response))
                st.write(f"LLM: {response}")
            except Exception as e:
                st.error(f"Error during chat: {e}")
                

if __name__ == "__main__":
    if "embed_model" not in st.session_state:
        embeddings = HuggingFaceEmbedding(model_name="meta-llama/Llama-3.2-1B")
        st.session_state["embed_model"] = embeddings
    
    if "llm" not in st.session_state:    
        llm = HuggingFaceLLM(model_name="meta-llama/Llama-3.2-1B",
                                    tokenizer_name="meta-llama/Llama-3.2-1B" )
        st.session_state["llm"] = llm
    main()