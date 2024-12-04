import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

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
from llama_index.core import SQLDatabase, PromptTemplate
from llama_index.core.prompts import PromptType
from sqlalchemy import create_engine
from llama_index.core.query_engine import NLSQLTableQueryEngine

from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.tools import QueryEngineTool


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


def build_sql_index():
    # Load structured data into SQLite database
    transaction_data = pd.read_csv("data/transaction_scores_processed.csv")
    # accountdoc_data = pd.read_csv("data/account_scores_processed.csv")

    conn = sqlite3.connect("SCORES.db")

    transaction_data.to_sql("transaction_score", conn, if_exists="replace", index=False)
    # accountdoc_data.to_sql("accountdoc_score", conn, if_exists="replace", index=False)

    print("Structured data loaded into SQLite database.")
    
    engine = create_engine("sqlite:///SCORES.db") 

    sql_db = SQLDatabase(engine, include_tables=["transaction_score"], view_support=True) 
    
    sql_query_engine = NLSQLTableQueryEngine(
    sql_database=sql_db,
    tables=["transaction_score"],
    embed_model="local",
    # text_to_sql_prompt=TEXT_TO_SQL_PROMPT,
    verbose=True
    )
    
    return sql_query_engine


# def get_sql_tool(sql_query_engine):
#     from llama_index.core.tools import QueryEngineTool

#     sql_tool = QueryEngineTool.from_defaults(
#         query_engine=sql_query_engine,
#         description=(
#             "Useful for translating a natural language query into a SQL query over"
#             " a table namely transaction_score "
#             " The table contain the risk scores of transactions."
#             " There are blended, ai, stat and rules risk scores along with conrol deviation " 
#             "and monitoring deviation reasons for each transaction."
#         ),
#     )
    
#     return sql_tool

        
def main():
    # Settings.embed_model = st.session_state["embed_model"]
    Settings.llm = st.session_state["llm"]
    st.title("LLM Chat Interface with File Uploads")

    # File uploads
    csv_file = st.file_uploader("Upload CSV files", type="csv")

    # Process and store files
    if csv_file:
        st.write("Processing CSV file...")
        if not os.path.exists("data"):
            os.makedirs("data")
        
        try:
            csv_path = save_uploaded_file(csv_file, "data")
            preprocess_csv(csv_path)
        except Exception as e:
            st.error(f"Error processing {csv_file.name}: {e}")

    # Button to create database and vector store
    if st.button("Create Database and Vector Store"):
        sql_query_engine = build_sql_index()
        st.session_state["query_engine"] = sql_query_engine

    if "query_engine" in st.session_state:
        st.write("### Chat with LLM")
        user_input = st.text_input("You:", "")

        if "messages" not in st.session_state:
            st.session_state["messages"] = []

        if user_input:
            try:
                response = st.session_state["query_engine"].query(user_input)
                print(response)
                st.session_state["messages"].append((user_input, response))
                st.write(f"LLM: {response}")
            except Exception as e:
                st.error(f"Error during chat: {e}")
                

if __name__ == "__main__":
    # if "embed_model" not in st.session_state:
    #     embeddings = HuggingFaceEmbedding(model_name="Qwen/Qwen2.5-1.5B-Instruct", trust_remote_code=True)
    #     st.session_state["embed_model"] = embeddings
    
    if "llm" not in st.session_state:    
        llm = HuggingFaceLLM(model_name="Qwen/Qwen2.5-1.5B-Instruct",
                                    tokenizer_name="Qwen/Qwen2.5-1.5B-Instruct")
    
        st.session_state["llm"] = llm
    main()