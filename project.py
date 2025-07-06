# project.py

import os
import faiss
import numpy as np
import streamlit as st
import docx2txt

from typing import List, Tuple
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

# Configuração da OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
EMBEDDING_MODEL = "text-embedding-ada-002"
CHAT_MODEL = "gpt-4"

# Lê documentos .docx
@st.cache_data(show_spinner="Extraindo textos dos documentos...")
def carregar_textos():
    documentos = []
    pasta = "support"
    for arquivo in os.listdir(pasta):
        if arquivo.endswith(".docx"):
            texto = docx2txt.process(os.path.join(pasta, arquivo))
            documentos.append((arquivo, texto.strip()))
    return documentos

# Gera embeddings com OpenAI v1.x
def gerar_embeddings(textos: List[str]):
    response = client.embeddings.create(
        input=textos,
        model=EMBEDDING_MODEL
    )
    return [r.embedding for r in response.data]

# Cria índice vetorial com FAISS
@st.cache_resource(show_spinner="Indexando documentos com FAISS...")
def criar_faiss_index(documentos):
    trechos = []
    referencias = []

    for nome, texto in documentos:
        for par in texto.split("\n"):
            par = par.strip()
            if len(par) > 30:
                trechos.append(par)
                referencias.append(nome)

    embeddings = gerar_embeddings(trechos)
    index = faiss.IndexFlatL2(len(embeddings[0]))
    index.add(np.array(embeddings).astype("float32"))

    return index, trechos, referencias, embeddings

# Busca os trechos mais relevantes
def buscar_contexto(pergunta: str, index, trechos, referencias, k=3) -> List[Tuple[str, str]]:
    pergunta_emb = gerar_embeddings([pergunta])[0]
    D, I = index.search(np.array([pergunta_emb]).astype("float32"), k)
    resultados = []
    for idx in I[0]:
        resultados.append((referencias[idx], trechos[idx]))
    return resultados

# Gera resposta com base nos trechos e retorna também o prompt
def gerar_resposta(pergunta: str, contexto: List[Tuple[str, str]]) -> Tuple[str, str]:
    prompt = f"""
Você é um assistente de atendimento ao cliente. Use as informações abaixo para responder de forma objetiva, amigável e precisa:

Contexto:
{chr(10).join(f'- {c}' for _, c in contexto)}

Pergunta: {pergunta}

Resposta:
"""

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=800
    )

    return response.choices[0].message.content, prompt

# ================================
#        INTERFACE STREAMLIT
# ================================

st.set_page_config(page_title="FAQ com OpenAI + FAISS", layout="centered")
st.title("🤖 FAQ Inteligente com OpenAI + FAISS")

# Inicializa histórico
if "mensagens" not in st.session_state:
    st.session_state.mensagens = []

# Carrega e indexa os documentos
documentos = carregar_textos()
index, trechos, refs, _ = criar_faiss_index(documentos)

# Entrada do usuário
pergunta = st.chat_input("Digite sua pergunta sobre o produto ou serviço:")

# Exibe histórico anterior
for msg in st.session_state.mensagens:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Processa nova pergunta
if pergunta:
    st.chat_message("user").markdown(pergunta)

    # Recupera contexto
    contexto = buscar_contexto(pergunta, index, trechos, refs)
    
    # Gera resposta e salva prompt usado
    resposta, prompt_usado = gerar_resposta(pergunta, contexto)

    # Exibe resposta + fontes + prompt
    with st.chat_message("assistant"):
        st.markdown(resposta)

        st.markdown("---")
        st.markdown("#### 📂 Documentos Recuperados:")
        for fonte, trecho in contexto:
            st.markdown(f"- **Arquivo:** `{fonte}`")
            st.markdown(f"  > {trecho.strip()[:300]}...")
            st.markdown("")

        st.markdown("#### 🧠 Prompt enviado ao modelo:")
        with st.expander("Visualizar prompt completo"):
            st.code(prompt_usado.strip())

    # Atualiza histórico
    st.session_state.mensagens.append({"role": "user", "content": pergunta})
    st.session_state.mensagens.append({"role": "assistant", "content": resposta})
