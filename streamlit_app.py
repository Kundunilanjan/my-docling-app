# streamlit_docling_rag.py
import streamlit as st
import os
import json
from pathlib import Path
from tempfile import mkdtemp

from langchain_docling import DoclingLoader, ExportType
from docling.chunking import HybridChunker
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_milvus import Milvus
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate

# Setup
os.environ["TOKENIZERS_PARALLELISM"] = "false"
st.set_page_config(page_title="Docling RAG", layout="wide")
st.title("📄 Ask Questions on Documents with Docling + LangChain")

# User inputs
file_url = st.text_input("Enter document URL (e.g., https://arxiv.org/pdf/2408.09869):")
question = st.text_input("Ask a question about the document:")
run_button = st.button("Run RAG")

if run_button and file_url and question:
    with st.spinner("🔍 Loading and chunking document..."):
        loader = DoclingLoader(
            file_path=[file_url],
            export_type=ExportType.DOC_CHUNKS,
            chunker=HybridChunker(tokenizer="sentence-transformers/all-MiniLM-L6-v2"),
        )
        docs = loader.load()
        splits = docs

    with st.spinner("🧠 Generating embeddings and building vector DB..."):
        embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        milvus_uri = str(Path(mkdtemp()) / "docling.db")
        vectorstore = Milvus.from_documents(
            documents=splits,
            embedding=embedding,
            collection_name="docling_demo",
            connection_args={"uri": milvus_uri},
            index_params={"index_type": "FLAT"},
            drop_old=True,
        )

    with st.spinner("🤖 Querying with RAG pipeline..."):
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        llm = HuggingFaceEndpoint(
            repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
            huggingfacehub_api_token=os.getenv("HF_TOKEN"),
            task="text-generation",
        )
        prompt = PromptTemplate.from_template(
            "Context information is below.\n---------------------\n{context}\n---------------------\nGiven the context information and not prior knowledge, answer the query.\nQuery: {input}\nAnswer:\n"
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

