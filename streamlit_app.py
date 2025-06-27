# streamlit_docling_rag.py
import streamlit as st
import os
import tempfile

from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint

# App setup
st.set_page_config(page_title="Docling RAG (Python 3.12-Compatible)", layout="wide")
st.title("📄 Ask Questions on Documents (No Docling)")

# Hugging Face token (set this in your environment or Streamlit secrets)
HF_TOKEN = os.getenv("HF_TOKEN", "your_hf_token_here")

# User inputs
uploaded_file = st.file_uploader("Upload a PDF file")
question = st.text_input("Ask a question about the document:")
run_button = st.button("Run RAG")

if run_button and uploaded_file and question:
    with st.spinner("🔍 Reading and splitting the document..."):
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
            tmp_pdf.write(uploaded_file.read())
            tmp_path = tmp_pdf.name

        # Load and split
        loader = PyMuPDFLoader(tmp_path)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
        splits = splitter.split_documents(docs)

    with st.spinner("🧠 Generating embeddings and building FAISS vector store..."):
        embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(splits, embedding)

    with st.spinner("🤖 Querying with RAG pipeline..."):
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        llm = HuggingFaceEndpoint(
            repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
            task="text-generation",
            huggingfacehub_api_token=HF_TOKEN
        )
        prompt = PromptTemplate.from_template(
            "Context information is below.\n---------------------\n{context}\n---------------------\n"
            "Given the context information and not prior knowledge, answer the query.\nQuery: {input}\nAnswer:"
        )
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        resp = rag_chain.invoke({"input": question})

        def clip(text, maxlen=350):
            return text[:maxlen] + "..." if len(text) > maxlen else text

        st.success("✅ Answer Generated")
        st.markdown(f"**Q:** {resp['input']}")
        st.markdown(f"**A:** {clip(resp['answer'])}")

        st.markdown("### 🔗 Source Documents")
        for i, doc in enumerate(resp["context"]):
            st.markdown(f"**Source {i+1}**: {clip(doc.page_content)}")
