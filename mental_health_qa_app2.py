import streamlit as st
import json
import requests
import re
import shutil
import spacy
import torch
import nltk
import numpy as np
from datasets import load_dataset
from transformers import pipeline
from sentence_transformers import CrossEncoder
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import NLTKTextSplitter
from langchain.schema import Document
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Mental Health QA (RAG)", layout="wide")

nltk.download("punkt")

@st.cache_resource
def load_models():
    qa_model = pipeline("question-answering", model="deepset/roberta-base-squad2", device=0 if torch.cuda.is_available() else -1)
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/multi-qa-mpnet-base-dot-v1")
    return qa_model, reranker, embedding_model

qa_model, reranker, embedding_model = load_models()

@st.cache_resource
def load_data():
    dataset = load_dataset("json", data_files="SQUAD_DATA.json", field="data")["train"]
    docs = [Document(page_content=para["context"]) for sample in dataset for para in sample["paragraphs"]]
    splitter = NLTKTextSplitter(chunk_size=500, chunk_overlap=100)
    sentence_chunks = splitter.split_documents(docs)
    seen = set()
    unique_chunks = [doc for doc in sentence_chunks if not (doc.page_content in seen or seen.add(doc.page_content))]
    filtered_chunks = [doc for doc in unique_chunks if len(doc.page_content.split()) > 25 and "7-" not in doc.page_content]
    shutil.rmtree("./chroma_db", ignore_errors=True)
    vectorstore = Chroma.from_documents(filtered_chunks, embedding=embedding_model, persist_directory="./chroma_db")
    return vectorstore.as_retriever(search_kwargs={"k": 20})

retriever = load_data()

def compute_f1(predicted, truth):
    pred_tokens = set(predicted.lower().split())
    truth_tokens = set(truth.lower().split())
    common = pred_tokens & truth_tokens
    if not common:
        return 0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(truth_tokens)
    return 2 * (precision * recall) / (precision + recall)

def compute_exact_match(predicted, truth):
    return int(predicted.strip().lower() == truth.strip().lower())

st.markdown("## 🧠 Mental Health QA with RAG + Cross-Encoder")

question = st.text_input("🔍 Enter your question:")
if question:
    with st.spinner("Retrieving and answering..."):
        docs = retriever.get_relevant_documents(question)
        passages = [doc.page_content for doc in docs]
        scores = reranker.predict([(question, p) for p in passages])
        top_passages = [p for s, p in sorted(zip(scores, passages), reverse=True)][:3]
        final_context = " ".join(top_passages)
        result = qa_model(question=question, context=final_context)
        st.markdown(f"### 🤖 Answer: {result['answer']}")
        st.markdown("#### 📚 Top Passages Used:")
        for i, p in enumerate(top_passages):
            st.markdown(f"**Passage {i+1}:** {p}")
