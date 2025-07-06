# project.py

import os
import streamlit as st
import torch
import docx2txt

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.openai import OpenAI  

# Corrige compatibilidade com Streamlit
torch.classes.__path__ = []

# Configuração da página
st.set_page_config(page_title="FAQ Chatbot", page_icon="💬", layout="centered")

st.sidebar.title("FAQ Chatbot")
st.sidebar.markdown("Versão com RAG + OpenAI + Streamlit Cloud")

# Exibição inicial
st.title("FAQ Inteligente com LLM + RAG")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Como posso te ajudar?"}]

# OpenAI LLM
llm = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-4",  # ou "gpt-3.5-turbo"
    temperature=0.7
)

# Embeddings
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Indexador de documentos
@st.cache_resource()
def carregar_base_vetorial():
    with st.spinner("Indexando base de conhecimento..."):
        reader = SimpleDirectoryReader(input_dir="./support", recursive=True)
        docs = reader.load_data()
        Settings.llm = llm
        Settings.embed_model = embed_model
        return VectorStoreIndex.from_documents(docs)

index = carregar_base_vetorial()

# Inicializa mecanismo de chat (com condensação de perguntas)
if "chat_engine" not in st.session_state:
    st.session_state.chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)

# Entrada do usuário
if prompt := st.chat_input("Sua pergunta:"):
    st.session_state.messages.append({"role": "user", "content": prompt})

# Exibição do histórico
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Geração de resposta
if st.session_state.messages[-1]["role"] == "user":
    with st.chat_message("assistant"):
        with st.spinner("Buscando resposta..."):
            user_message = st.session_state.messages[-1]["content"]
            contextual_prompt = (
                f"Você é um atendente útil. A pergunta é: '{user_message}'. "
                "Use os documentos de suporte disponíveis para responder com precisão e clareza."
            )

            response = st.session_state.chat_engine.chat(contextual_prompt)

            st.markdown(response.response)

            # Fontes utilizadas
            if response.source_nodes:
                st.markdown("##### 🔎 Fontes:")
                for node in response.source_nodes:
                    if 'file_path' in node.metadata:
                        st.markdown(f"- **Arquivo**: `{node.metadata['file_path']}`")
                    st.markdown(f" Trecho: `{node.text.strip()[:300]}...`")

            st.session_state.messages.append({"role": "assistant", "content": response.response})
