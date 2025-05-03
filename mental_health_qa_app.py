import streamlit as st
import json
import requests
import re
import spacy
import torch
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize
import nltk

# ✅ FIRST Streamlit command
st.set_page_config(page_title="Mental Health QA", layout="wide")

# Ensure punkt is available
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

# -------------------------------
# Load Models and Components
# -------------------------------
@st.cache_resource
def load_models():
    qa_pipeline = pipeline(
        "question-answering",
        model="bert-large-uncased-whole-word-masking-finetuned-squad",
        device=0 if torch.cuda.is_available() else -1
    )
    sent_model = SentenceTransformer('all-MiniLM-L6-v2')
    nlp = spacy.load("en_core_web_sm")
    return qa_pipeline, sent_model, nlp

qa_pipeline_kg, sentence_model, nlp = load_models()

# -------------------------------
# Helper Functions
# -------------------------------
def clean_context(text):
    text = re.sub(r"PERSON|NAME|AT_MENTION|USER_ID", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def chunk_context(context, max_len=400):
    sentences = sent_tokenize(context)
    chunks, chunk = [], ""
    for sent in sentences:
        if len(chunk) + len(sent) < max_len:
            chunk += " " + sent
        else:
            chunks.append(chunk.strip())
            chunk = sent
    if chunk:
        chunks.append(chunk.strip())
    return chunks

def generate_sparql_query(question):
    doc = nlp(question.lower())
    mental_terms = [
        "anxiety", "depression", "stress", "bipolar", "ocd",
        "trauma", "insomnia", "ptsd", "adhd", "burnout", "panic"
    ]
    entity = None
    for term in mental_terms:
        if term in question.lower():
            entity = term
            break
    if not entity:
        for chunk in doc.noun_chunks:
            if chunk.text.lower() in mental_terms:
                entity = chunk.text
                break
    if entity:
        return f"""
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX schema: <http://schema.org/>

        SELECT ?entity ?label ?description WHERE {{
          ?entity rdfs:label "{entity}"@en.
          ?entity schema:description ?description.
          FILTER(LANG(?description) = "en")
        }}
        LIMIT 1
        """
    return None

def query_knowledge_graph(question):
    sparql_query = generate_sparql_query(question)
    if not sparql_query:
        return None
    response = requests.get(
        "https://query.wikidata.org/sparql",
        params={"query": sparql_query, "format": "json"},
        headers={"User-Agent": "Mozilla/5.0"}
    )
    if response.status_code == 200:
        bindings = response.json().get('results', {}).get('bindings', [])
        return bindings[0] if bindings else None
    return None

def answer_question_with_bert(question, context):
    try:
        result = qa_pipeline_kg(question=question, context=context)
        return result['answer']
    except:
        return ""

def qa_with_kg_fusion(question, context):
    kg_info = query_knowledge_graph(question)
    kg_answer = kg_info.get('description', {}).get('value', '') if kg_info else ''
    bert_answer = answer_question_with_bert(question, context)

    # Just combine KG info without checking similarity
    if kg_answer:
        final_answer = f"{bert_answer} (Related Info: {kg_answer})"
    else:
        final_answer = bert_answer

    return final_answer, bert_answer, kg_answer

# -------------------------------
# UI Styling and Layout
# -------------------------------
st.markdown("""
    <style>
    html, body, [class*="css"]  {
        font-size: 18px !important;
    }
    .stTextInput > div > div > input, .stTextArea textarea {
        font-size: 18px !important;
        padding: 10px 14px !important;
    }
    .stButton > button {
        font-size: 18px !important;
        padding: 10px 20px !important;
    }
    .stMarkdown {
        padding-top: 0.8em;
        padding-bottom: 0.8em;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='font-size: 36px;'>🧠 Mental Health QA System with KG Fusion</h1>", unsafe_allow_html=True)

# -------------------------------
# Input Widgets
# -------------------------------
question = st.text_input(
    "🔍 Ask a mental health-related question",
    placeholder="e.g., What causes anxiety at night?"
)

context = st.text_area(
    "📚 Provide the context for your question",
    placeholder="e.g., I feel my heart racing and can't stop thinking before bed.",
    height=250
)

# -------------------------------
# Answer Output
# -------------------------------
if st.button("Get Answer"):
    if not question.strip() or not context.strip():
        st.warning("Please enter both question and context.")
    else:
        with st.spinner("Thinking..."):
            context = clean_context(context)
            chunks = chunk_context(context)
            best_answer, best_kg = "", ""

            for chunk in chunks:
                try:
                    final, bert, kg = qa_with_kg_fusion(question, chunk)
                    if len(final) > len(best_answer):
                        best_answer = final
                        best_kg = kg
                except Exception as e:
                    print(f"Error: {e}")
                    continue

        st.success("Answer Ready")
        st.markdown(f"<strong>Answer:</strong> {best_answer}", unsafe_allow_html=True)
        if best_kg:
            st.markdown(f"<strong>Knowledge Graph Description:</strong> {best_kg}", unsafe_allow_html=True)
