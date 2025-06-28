import streamlit as st
import fitz  # PyMuPDF
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub

# Streamlit UI
st.set_page_config(page_title="PDF Q&A RAG App", layout="centered")
st.title("📄 PDF Question Answering with RAG")

# File Upload
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

# Extract text from PDF
@st.cache_data
def extract_documents(file):
    pdf = fitz.open(stream=file.read(), filetype="pdf")
    docs = []
    for i in range(len(pdf)):
        text = pdf.load_page(i).get_text()
        docs.append(Document(page_content=text, metadata={"page": i+1}))
    return docs

# Split and embed
@st.cache_resource
def prepare_vectorstore(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

if uploaded_file:
    with st.spinner("Processing PDF..."):
        docs = extract_documents(uploaded_file)
        vectorstore = prepare_vectorstore(docs)
        retriever = vectorstore.as_retriever()
        llm = HuggingFaceHub(
            repo_id="google/flan-t5-large",
            model_kwargs={"temperature": 0.1, "max_length": 512}
        )
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    st.success("PDF processed successfully!")

    # Question input
    question = st.text_input("Ask a question about your PDF:")
    if question:
        with st.spinner("Searching..."):
            response = qa_chain.run(question)
            st.write("### 📌 Answer:")
            st.write(response)

