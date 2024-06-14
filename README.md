# PDF-Insight: Streamlit Retrieval-Augmented Generation (RAG) 🗞️

![image](https://github.com/Armanasq/RAG-PDF-Streamlit/assets/60850934/9249dfb9-6b63-41bb-8518-bd58ff6f2a77)

This repository contains a Streamlit application designed for performing retrieval-augmented generation on PDF documents using large language models and embeddings from HuggingFace. 
The application allows users to upload a PDF, ingest its content into a vector store, and query the document using natural language.

## Features

- **PDF Upload and Display**: Upload and view PDF files within the application.
- **Data Ingestion**: Ingest PDF content into a persistent storage context for efficient querying.
- **Query Handling**: Perform natural language queries on the ingested PDF content.
- **Language Detection**: Detects the language of the query to ensure accurate processing.

## Setup

### Prerequisites

- Python 3.8+
- Streamlit
- HuggingFace Transformers
- Llama Index
- dotenv
- langdetect

### Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/Armanasq/RAG-PDF-Streamli.git
    cd your-repo
    ```

2. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Set up environment variables**:
    Create a `.env` file in the root directory and add your HuggingFace API token:
    ```env
    HUGGINGFACE_TOKEN=your_huggingface_api_token
    ```

## Usage

1. **Run the Streamlit application**:
    ```bash
    streamlit run streamlitRAG.py
    ```

2. **Upload a PDF**:
    - Use the file uploader in the Streamlit interface to upload a PDF.

3. **Ingest PDF Content**:
    - The content of the uploaded PDF will be ingested and stored for querying.

4. **Query the PDF Content**:
    - Enter a query related to the PDF content in the text input box.
    - The application will return relevant information based on the content of the PDF.

## Code Overview

### Environment Setup

- **Load Environment Variables**:
    ```python
    load_dotenv()
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    ```

### Llama Index Configuration

- **Configure Llama Index to use GPU and HuggingFace models**:
    ```python
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
    ```

### Data Handling

- **PDF Display Function**:
    ```python
    def display_pdf(file_path):
        with open(file_path, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
        return pdf_display
    ```

- **Data Ingestion**:
    ```python
    def data_ingestion():
        documents = SimpleDirectoryReader(DATA_DIR).load_data()
        storage_context = StorageContext.from_defaults()
        index = VectorStoreIndex.from_documents(documents)
        index.storage_context.persist(persist_dir=PERSIST_DIR)
    ```

### Query Handling

- **Handle Query Function**:
    ```python
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
    ```

### Streamlit UI

- **Streamlit Interface**:
    ```python
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
    ```

## Contributing

Contributions are welcome. Please fork the repository and submit a pull request for any feature requests or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
