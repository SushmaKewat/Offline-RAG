import time
import streamlit as st
from vanna_calls import (
    generate_questions_cached,
    generate_sql_cached,
    run_sql_cached,
    is_sql_valid_cached,
    generate_summary_cached
)
from sqlalchemy import create_engine, inspect, text
from llama_index.core import SQLDatabase
import sqlite3
import pandas as pd

st.set_page_config(layout="wide")

st.sidebar.title("Output Settings")
st.sidebar.checkbox("Show SQL", value=True, key="show_sql")
st.sidebar.checkbox("Show Table", value=True, key="show_table")
st.sidebar.checkbox("Show Summary", value=True, key="show_summary")
st.sidebar.button("Reset", on_click=lambda: set_question(None), use_container_width=True)

st.title("Chat with your SQL Database")
# st.sidebar.write(st.session_state)

def get_table_data(table_name, conn):
            query = f"SELECT * FROM {table_name}"
            df = pd.read_sql_query(query, conn)
            return df
        
def load_db():
    # Load the SQLite database
    engine = create_engine("sqlite:///SCORES.db?mode=ro", connect_args={"uri": True})

    sql_database = SQLDatabase(engine) #include all tables
    
    return sql_database, engine

def add_to_message_history(role, content):
            message = {"role": role, "content": str(content)}
            st.session_state["messages"].append(
                message
            ) 


def set_question(question):
    my_question = question
    
sql_database, engine = load_db()

# Create an inspector object
inspector = inspect(engine)

# Get list of tables in the database
table_names = inspector.get_table_names()

# Sidebar selection for tables
selected_table = st.sidebar.selectbox("Select a Table", table_names)

db_file = 'SCORES.db'
conn = sqlite3.connect(db_file)

# Display the selected table
if selected_table:
    df = get_table_data(selected_table, conn)
    st.sidebar.text(f"Data for table '{selected_table}':")
    st.sidebar.dataframe(df)

# Close the connection
conn.close()

# assistant_message_suggested = st.chat_message(
#     "assistant", 
#     # avatar=avatar_url
# )
# if assistant_message_suggested.button("Click to show suggested questions"):
#     st.session_state["my_question"] = None
#     questions = generate_questions_cached()
#     for i, question in enumerate(questions):
#         time.sleep(0.05)
#         button = st.button(
#             question,
#             on_click=set_question,
#             args=(question,),
#         )

if "messages" not in st.session_state:  # Initialize the chat messages history
    st.session_state["messages"] = [
        {"role": "assistant", "content": f"Hello. Ask me anything related to the database."}
    ]
# my_question = st.session_state.get("my_question", default=None)

for message in st.session_state["messages"]:  # Display the prior chat messages
    with st.chat_message(message["role"]):
        print(message["content"])
        st.write(message["content"])
# if my_question is None:
my_question = st.chat_input(
        "Ask me a question about your data",
    )


if my_question:
    # st.session_state["my_question"] = my_question
    with st.chat_message("user"):
        st.write(f"{my_question}")
    add_to_message_history("user", my_question)

    sql = generate_sql_cached(question=my_question+" in the table "+selected_table)

    if sql:
        if is_sql_valid_cached(sql=sql):
            if st.session_state.get("show_sql", True):
                assistant_message_sql = st.chat_message(
                    "assistant", 
                    # avatar=avatar_url
                )
                assistant_message_sql.code(sql, language="sql", line_numbers=True)
                add_to_message_history("assistant", sql)
        else:
            assistant_message = st.chat_message(
                "assistant", 
                # avatar=avatar_url
            )
            assistant_message.write(sql)
            st.stop()

        df = run_sql_cached(sql=sql)

        if df is not None:
            st.session_state["df"] = df

        if st.session_state.get("df") is not None:
            if st.session_state.get("show_table", True):
                df = st.session_state.get("df")
                assistant_message_table = st.chat_message(
                    "assistant",
                    # avatar=avatar_url,
                )
                if len(df) > 10:
                    assistant_message_table.text("First 10 rows of data")
                    assistant_message_table.dataframe(df.head(10))
                else:
                    assistant_message_table.dataframe(df)
                add_to_message_history("assistant", df)


            if st.session_state.get("show_summary", True):
                assistant_message_summary = st.chat_message(
                    "assistant",
                    # avatar=avatar_url,
                )
                summary = generate_summary_cached(question=my_question, df=df)
                if summary is not None:
                    assistant_message_summary.text(summary)
                    add_to_message_history("assistant", summary)

    else:
        assistant_message_error = st.chat_message(
            "assistant", 
            # avatar=avatar_url
        )
        assistant_message_error.error("I wasn't able to generate SQL for that question")
    my_question = None