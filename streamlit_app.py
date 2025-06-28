import streamlit as st
import fitz  # PyMuPDF
from PIL import Image
from langchain.schema import Document
import io
import hashlib

st.set_page_config(page_title="📄 PDF Extractor", layout="wide")
st.title("🔍 PDF Extractor: Text, Metadata, Images, Links")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

def extract_pdf_data(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    extracted_docs = []
    all_images = {}
    page_data = []

    for i in range(len(doc)):
        page = doc.load_page(i)
        text = page.get_text()
        links = page.get_links()
        images = page.get_images(full=True)
        metadata = {
            "page": i + 1,
            "links": [l['uri'] for l in links if l.get('uri')],
            "num_images": len(images)
        }

        # Text + Metadata as LangChain Document
        extracted_docs.append(Document(page_content=text, metadata=metadata))

        # Extract unique images
        for img_index, img in enumerate(images):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            img_hash = hashlib.md5(image_bytes).hexdigest()
            if img_hash not in all_images:
                all_images[img_hash] = image_bytes

        page_data.append({
            "text": text,
            "links": metadata["links"],
            "num_images": metadata["num_images"],
        })

    return extracted_docs, all_images, page_data

if uploaded_file:
    with st.spinner("Extracting PDF content..."):
        docs, images, page_data = extract_pdf_data(uploaded_file)
    st.success("Extraction complete!")

    with st.expander("📄 Page-wise Text, Metadata & Links"):
        for idx, data in enumerate(page_data):
            st.markdown(f"### 📃 Page {idx + 1}")
            st.write("📝 Text:")
            st.write(data["text"] or "*No text found*")
            st.write(f"🔗 Links ({len(data['links'])}):")
            for link in data["links"]:
                st.markdown(f"- [{link}]({link})", unsafe_allow_html=True)
            st.write(f"🖼️ Number of images: {data['num_images']}")
            st.markdown("---")

    if images:
        st.markdown("## 🖼️ Extracted Unique Images")
        cols = st.columns(3)
        for i, (hash_id, img_data) in enumerate(images.items()):
            img = Image.open(io.BytesIO(img_data))
            cols[i % 3].image(img, caption=f"Image {i+1}", use_column_width="always")
    else:
        st.info("No images found in the PDF.")
