# streamlit_docling_rag.py
import streamlit as st
import os
import tempfile
import fitz  # PyMuPDF

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint

# Set up app
st.set_page_config(page_title="📄 PDF RAG (Python 3.12)", layout="wide")
st.title("📄 Ask Questions on Uploaded PDF (RAG)")

# Hugging Face Token (set as env var or replace below)
HF_TOKEN = os.getenv("HF_TOKEN", "your_hf_token_here")

# Upload + Question
uploaded_file = st.file_uploader("📤 Upload a PDF file", type="pdf")
question = st.text_input("❓ Ask a question from the document:")
run_button = st.button("Run RAG")

# Function to load PDF using fitz directly
def load_pdf_with_fitz(file_path):
    doc = fitz.open(file_path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text()
        if text.strip():  # skip blank pages
            pages.append(Document(page_content=text, metadata={"page": i + 1}))
    return pages

# Run pipeline
if run_button and uploaded_file and question:
    with st.spinner("📖 Reading and splitting the document..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        docs = load_pdf_with_fitz(tmp_path)
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
        chunks = splitter.split_documents(docs)

    with st.spinner("🧠 Building vector store..."):
        embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(chunks, embedder)

    with st.spinner("🤖 Generating answer..."):
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        llm = HuggingFaceEndpoint(
            repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
            huggingfacehub_api_token=HF_TOKEN,
            task="text-generation"
        )

        prompt = PromptTemplate.from_template(
            "Context:\n---------------------\n{context}\n---------------------\n"
            "Answer the question based only on the above context.\n"
            "Question: {input}\nAnswer:"
        )

        qa_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, qa_chain)

        result = rag_chain.invoke({"input": question})

        def clip(text, maxlen=500):
            return text[:maxlen] + "..." if len(text) > maxlen else text

        st.success("✅ Answer:")
        st.markdown(f"**Q:** {result['input']}")
        st.markdown(f"**A:** {clip(result['answer'])}")

        st.markdown("### 🔍 Source Snippets:")
        for i, doc in enumerate(result["context"]):
            st.markdown(f"**Snippet {i+1}** (Page {doc.metadata.get('page')}): {clip(doc.page_content)}")

