# project.py

import os
import faiss
import numpy as np
import streamlit as st
import docx2txt

from typing import List, Tuple
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

# OpenAI Configuration
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
EMBEDDING_MODEL = "text-embedding-ada-002"
CHAT_MODEL = "gpt-4"

# Load .docx documents
@st.cache_data(show_spinner="Extracting text from documents...")
def load_documents():
    documents = []
    folder = "support"
    for filename in os.listdir(folder):
        if filename.endswith(".docx"):
            text = docx2txt.process(os.path.join(folder, filename))
            documents.append((filename, text.strip()))
    return documents

# Generate embeddings using OpenAI v1.x
def generate_embeddings(texts: List[str]):
    response = client.embeddings.create(
        input=texts,
        model=EMBEDDING_MODEL
    )
    return [r.embedding for r in response.data]

# Create FAISS vector index
@st.cache_resource(show_spinner="Indexing documents with FAISS...")
def create_faiss_index(documents):
    passages = []
    sources = []

    for name, text in documents:
        for paragraph in text.split("\n"):
            paragraph = paragraph.strip()
            if len(paragraph) > 30:
                passages.append(paragraph)
                sources.append(name)

    embeddings = generate_embeddings(passages)
    index = faiss.IndexFlatL2(len(embeddings[0]))
    index.add(np.array(embeddings).astype("float32"))

    return index, passages, sources, embeddings

# Retrieve the most relevant passages
def retrieve_context(query: str, index, passages, sources, k=3) -> List[Tuple[str, str]]:
    query_emb = generate_embeddings([query])[0]
    D, I = index.search(np.array([query_emb]).astype("float32"), k)
    results = []
    for idx in I[0]:
        results.append((sources[idx], passages[idx]))
    return results

# Generate answer using the retrieved context and return prompt as well
def generate_answer(query: str, context: List[Tuple[str, str]]) -> Tuple[str, str]:
    prompt = f"""
You are a customer support assistant. Use the information below to answer in a clear, friendly, and helpful way.

Context:
{chr(10).join(f'- {c}' for _, c in context)}

Question: {query}

Answer:
"""

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=800
    )

    return response.choices[0].message.content, prompt

# ================================
#         STREAMLIT UI
# ================================

st.set_page_config(page_title="FAQ with OpenAI + FAISS", layout="centered")
st.title("🤖 Smart FAQ with OpenAI + FAISS")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Load and index documents
documents = load_documents()
index, passages, refs, _ = create_faiss_index(documents)

# User input
query = st.chat_input("Type your question about the product or service:")

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Handle new query
if query:
    st.chat_message("user").markdown(query)

    # Retrieve relevant context
    context = retrieve_context(query, index, passages, refs)
    
    # Generate response and capture prompt
    answer, used_prompt = generate_answer(query, context)

    # Show response + sources + prompt
    with st.chat_message("assistant"):
        st.markdown(answer)

        st.markdown("---")
        st.markdown("#### 📂 Retrieved Documents:")
        for source, passage in context:
            st.markdown(f"- **File:** `{source}`")
            st.markdown(f"  > {passage.strip()[:300]}...")
            st.markdown("")

        st.markdown("#### 🧠 Prompt Sent to the Model:")
        with st.expander("Show full prompt"):
            st.code(used_prompt.strip())

    # Update chat history
    st.session_state.messages.append({"role": "user", "content": query})
    st.session_state.messages.append({"role": "assistant", "content": answer})
