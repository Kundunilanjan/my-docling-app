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

# Set Hugging Face Token (use secrets or env vars in production)
HF_TOKEN = os.getenv("HF_TOKEN", "your_hf_token_here")

# Streamlit config
st.set_page_config(page_title="📄 RAG on PDF", layout="wide")
st.title("📄 Ask Questions on Uploaded PDF")

# Upload & input
uploaded_file = st.file_uploader("📤 Upload a PDF", type="pdf")
question = st.text_input("❓ Ask a question about the document:")
run_button = st.button("🔍 Run RAG")

# Display toggles
show_pages = st.checkbox("📃 Show full PDF pages")
show_chunks = st.checkbox("🧩 Show chunked text")

# PDF loader
def load_pdf(file_path):
    doc = fitz.open(file_path)
    return [
        Document(page_content=page.get_text(), metadata={"page": i + 1})
        for i, page in enumerate(doc)
        if page.get_text().strip()
    ]

# Clipping long text
def clip(text, maxlen=500):
    return text[:maxlen] + "..." if len(text) > maxlen else text

# On Run
if run_button:
    if not uploaded_file or not question:
        st.warning("⚠️ Upload a PDF and ask a question.")
        st.stop()

    with st.spinner("📖 Reading and splitting the document..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        st.success(f"✅ Uploaded: `{uploaded_file.name}`")
        docs = load_pdf(tmp_path)

        if not docs:
            st.error("❌ No readable text found in PDF.")
            st.stop()

        if show_pages:
            st.subheader("📃 Extracted Pages")
            for d in docs:
                st.markdown(f"**Page {d.metadata['page']}**\n```\n{clip(d.page_content)}\n```")

        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
        chunks = splitter.split_documents(docs)

        if show_chunks:
            st.subheader("🧩 Split Text Chunks")
            for i, chunk in enumerate(chunks[:5]):
                st.markdown(f"**Chunk {i+1}**\n```\n{clip(chunk.page_content)}\n```")

    with st.spinner("🔗 Embedding and indexing..."):
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(chunks, embeddings)

    with st.spinner("🤖 Generating answer..."):
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        llm = HuggingFaceEndpoint(
            repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
            task="text-generation",
            huggingfacehub_api_token=HF_TOKEN
        )

        prompt = PromptTemplate.from_template(
            "Context:\n---------------------\n{context}\n---------------------\n"
            "Answer the question strictly based on the context above.\n"
            "Question: {input}\nAnswer:"
        )

        qa_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, qa_chain)

        try:
            result = rag_chain.invoke({"input": question})
        except Exception as e:
            st.error(f"❌ Error during LLM inference:\n\n{e}")
            st.stop()

        st.success("✅ Answer Generated")
        st.markdown(f"**🧠 Q:** {result['input']}")
        st.markdown(f"**💬 A:** {clip(result['answer'])}")

        st.subheader("📚 Source Contexts")
        for i, doc in enumerate(result["context"]):
            page = doc.metadata.get("page", "?")
            st.markdown(f"**Snippet {i+1}** (Page {page})\n> {clip(doc.page_content)}")
