
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

# Hugging Face Token (set as environment variable or replace here)
HF_TOKEN = os.getenv("HF_TOKEN", "your_hf_token_here")

# Streamlit UI setup
st.set_page_config(page_title="📄 PDF Q&A with RAG", layout="wide")
st.title("📄 Ask Questions on Uploaded PDF")

# File and question input
uploaded_file = st.file_uploader("📤 Upload a PDF", type="pdf")
question = st.text_input("❓ Ask a question from the document:")
run_button = st.button("🔎 Run RAG")

# Optional toggles
show_pages = st.checkbox("📃 Show extracted PDF pages", value=False)
show_chunks = st.checkbox("🧩 Show split text chunks", value=False)


# Function to read PDF using fitz (PyMuPDF)
def load_pdf_with_fitz(file_path):
    doc = fitz.open(file_path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text()
        if text.strip():  # skip blank pages
            pages.append(Document(page_content=text, metadata={"page": i + 1}))
    return pages


# Run pipeline when button clicked
if run_button:
    if not uploaded_file or not question:
        st.warning("⚠️ Please upload a PDF and enter a question.")
        st.stop()

    with st.spinner("📖 Extracting text from PDF..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        st.markdown(f"✅ Uploaded File: `{uploaded_file.name}`")
        docs = load_pdf_with_fitz(tmp_path)

        if show_pages:
            st.markdown("### 📃 Extracted Pages:")
            for i, doc in enumerate(docs):
                st.markdown(f"**Page {doc.metadata['page']}**:\n```\n{doc.page_content[:500]}\n```")

        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
        chunks = splitter.split_documents(docs)

        if show_chunks:
            st.markdown("### 🧩 Text Chunks:")
            for i, chunk in enumerate(chunks[:5]):
                st.markdown(f"**Chunk {i+1}**:\n```\n{chunk.page_content[:500]}\n```")

    with st.spinner("🔗 Embedding and indexing text..."):
        embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(chunks, embedder)

    with st.spinner("🤖 Generating answer from RAG..."):
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

        try:
            result = rag_chain.invoke({"input": question})
        except Exception as e:
            st.error(f"❌ Error during LLM inference: {e}")
            st.stop()

        def clip(text, maxlen=500):
            return text[:maxlen] + "..." if len(text) > maxlen else text

        st.success("✅ Answer:")
        st.markdown(f"**Q:** {result['input']}")
        st.markdown(f"**A:** {clip(result['answer'])}")

        st.markdown("### 📚 Source Snippets Used:")
        for i, doc in enumerate(result["context"]):
            page = doc.metadata.get("page", "?")
            st.markdown(f"**Snippet {i+1}** (Page {page}):\n> {clip(doc.page_content)}")
