import streamlit as st
import os
import base64
from llama_index.core import StorageContext, load_index_from_storage, VectorStoreIndex, SimpleDirectoryReader, ChatPromptTemplate, Settings
from llama_index.llms.huggingface import HuggingFaceInferenceAPI
from dotenv import load_dotenv
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from langdetect import detect
import subprocess

# Load environment variables
load_dotenv()

# Setup HuggingFace API token from environment
hf_token = os.getenv("HUGGINGFACE_TOKEN")

# Configure Llama Index settings to use GPU (if available)
Settings.llm = HuggingFaceInferenceAPI(
    model_name="mistralai/Mistral-7B-Instruct-v0.3",
    tokenizer_name="mistralai/Mistral-7B-Instruct-v0.3",
    context_window=5000,
    max_new_tokens=1024,
    generate_kwargs={"temperature": 0.1},
    device=0  # Use GPU 0, set to -1 for CPU
)
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5",
    device=0  # Use GPU 0, set to -1 for CPU
)

# Define the directory for persistent storage and data
PERSIST_DIR = "./db"
DATA_DIR = "data"

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PERSIST_DIR, exist_ok=True)

def display_pdf(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
    return pdf_display

def data_ingestion():
    documents = SimpleDirectoryReader(DATA_DIR).load_data()
    storage_context = StorageContext.from_defaults()
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=PERSIST_DIR)

def handle_query(query, lang):
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)
    chat_text_qa_msgs = [
    (
        "user",
        """You are a Q&A assistant. Your main goal is to provide answers as accurately as possible, based on the instructions and context you have been given. If a question does not match the provided context or is outside the scope of the document, kindly advise the user to ask questions within the context of the document.
        Context:
        {context_str}
        Question:
        {query_str}
        """
    )
    ]
    text_qa_template = ChatPromptTemplate.from_messages(chat_text_qa_msgs)
    
    query_engine = index.as_query_engine(text_qa_template=text_qa_template)
    answer = query_engine.query(query)
    
    if hasattr(answer, 'response'):
        return answer.response, lang
    elif isinstance(answer, dict) and 'response' in answer:
        return answer['response'], lang
    else:
        return "Sorry, I couldn't find an answer.", lang

def process_file(uploaded_file):
    if uploaded_file:
        filepath = os.path.join(DATA_DIR, "saved_pdf.pdf")
        with open(filepath, "wb") as f:
            f.write(uploaded_file.getbuffer())
        data_ingestion()
        return display_pdf(filepath)
    return "No file uploaded."

# Streamlit UI
st.title("(PDF) Information and Inference 🗞️")
st.markdown("## Retrieval-Augmented Generation")
st.markdown("Start chat ...🚀")

uploaded_file = st.file_uploader("Upload your PDF file", type="pdf")
query = st.text_input("Ask me anything about the content of the PDF:")
chat_history = st.empty()

if uploaded_file:
    pdf_display = process_file(uploaded_file)
    st.markdown(pdf_display, unsafe_allow_html=True)

if query:
    lang = detect(query)
    response, lang = handle_query(query, lang)
    chat_history.text_area("Chat History", f"User: {query}\nAssistant: {response}", height=300)
