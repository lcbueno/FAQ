# Personalized Chatbot with Recommendation System for Customer Support Using LLMs

import os
import docx2txt
import streamlit as st
import torch

from dotenv import load_dotenv  
load_dotenv()                   

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings


# Fixes compatibility between Streamlit and PyTorch
torch.classes.__path__ = []

# Page configuration
st.set_page_config(page_title="FAQ Project", page_icon=":MVP", layout="centered")

# Sidebar menu
st.sidebar.title("FAQ Project")
st.sidebar.markdown("### Customer Service")
st.sidebar.markdown("[thechaincademy](https://www.linkedin.com/company/thechaincademy/posts/?feedView=all)")
st.sidebar.button("AI Chatbot Version 1.0")

# Main title
st.title("Personalized Chatbot with Recommendation System for Customer Support Using LLMs")

# Message history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Enter your question"}]

# LLM via OpenAI
llm = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),  # configure como variável secreta no Streamlit Cloud
    model="gpt-4",                         # ou "gpt-3.5-turbo"
    temperature=0.7
)

# Embeddings
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Vector database
@st.cache_resource()
def dsa_cria_database_externo():
    with st.spinner(text="Loading and indexing documents. This should take a few seconds."):
        reader = SimpleDirectoryReader(input_dir="./support", recursive=True)
        docs = reader.load_data()
        Settings.llm = llm
        Settings.embed_model = embed_model
        index = VectorStoreIndex.from_documents(docs)
        return index

banco_vetorial = dsa_cria_database_externo()

# Initialize chat engine
if "chat_engine" not in st.session_state:
    st.session_state.chat_engine = banco_vetorial.as_chat_engine(chat_mode="condense_question", verbose=True)

# User input
if prompt := st.chat_input("Your question"):
    st.session_state.messages.append({"role": "user", "content": prompt})

# Chat history display
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Generate response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            user_message = st.session_state.messages[-1]["content"]
            contextual_prompt = (
                f"You are a dedicated support agent. The user asked the following question: '{user_message}'. "
                "Consider all available Q&A documents and provide a detailed and accurate response, "
                "making recommendations where appropriate. Be proactive."
            )

            response = st.session_state.chat_engine.chat(contextual_prompt)

             # Display main answer
            st.write(response.response)

            # Display sources
            if response.source_nodes:
                st.markdown("#### ⭐️ Source:")
                for node in response.source_nodes:
                    if 'file_path' in node.metadata:
                        st.markdown(f"**Documento:** `{node.metadata['file_path']}`")
                    st.markdown(f"**Trecho:**\n> {node.text.strip()[:500]}...")

            st.session_state.messages.append({"role": "assistant", "content": response.response})
