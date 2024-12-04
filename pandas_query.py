from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core import Settings

import streamlit as st 
import os
import pandas as pd
from llama_index.experimental.query_engine import PandasQueryEngine


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

def main():
    # Settings.embed_model = st.session_state["embed_model"]
    Settings.llm = st.session_state["llm"]
    st.title("LLM Chat Interface with File Uploads")

    # File uploads
    csv_file = st.file_uploader("Upload CSV file", type="csv")

    # Process and store files
    if csv_file:
        st.write("Processing CSV files...")
        if not os.path.exists("data"):
            os.makedirs("data")
        
        try:
            csv_path = save_uploaded_file(csv_file, "data")
            csv_file_path = preprocess_csv(csv_path)
        except Exception as e:
            st.error(f"Error processing {csv_file.name}: {e}")

    # Button to create database and vector store
    if st.button("Start chatting"):
        print(csv_file)
        df = pd.read_csv("data/transaction_scores_processed.csv")
        query_engine = PandasQueryEngine(df=df, verbose=True ,synthesize_response=True)
        st.session_state["query_engine"] = query_engine

    if "query_engine" in st.session_state:
        st.write("### Chat with LLM")

        if "messages" not in st.session_state:
            st.session_state["messages"] = []
            
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                       
        if prompt := st.chat_input("Ask questions about your data..."):
            try:
                # Display user message in chat message container
                st.chat_message("user").markdown(prompt)
                # Add user message to chat history
                st.session_state.messages.append({"role": "user", "content": prompt})

                response = st.session_state["query_engine"].query(prompt)
                # Display assistant response in chat message container
                with st.chat_message("assistant"):
                    st.markdown(response)
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content":(response)})
            except Exception as e:
                st.error(f"Error during chat: {e}")

        # if user_input:
        #     try:
        #         response = st.session_state["query_engine"].query(user_input)
        #         print(response)
        #         st.session_state["messages"].append((user_input, response))
        #         st.write(f"LLM: {response}")
        #     except Exception as e:
        #         st.error(f"Error during chat: {e}")
                

if __name__ == "__main__":
    # if "embed_model" not in st.session_state:
    #     embeddings = HuggingFaceEmbedding(model_name="Qwen/Qwen2.5-1.5B-Instruct", trust_remote_code=True)
    #     st.session_state["embed_model"] = embeddings
    
    if "llm" not in st.session_state:    
        llm = HuggingFaceLLM(model_name="Qwen/Qwen2.5-1.5B-Instruct",
                                    tokenizer_name="Qwen/Qwen2.5-1.5B-Instruct")
    
        st.session_state["llm"] = llm
    main()