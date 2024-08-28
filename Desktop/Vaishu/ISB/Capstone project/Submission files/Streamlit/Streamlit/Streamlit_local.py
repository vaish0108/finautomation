import streamlit as st
import doc_extractor
import pdf_extractor
import pdf_ocr_extractor
import xlcsv_extractor
import image_extractor

st.set_page_config(layout="wide")

st.markdown("""
    <div style='
        background-color: #f0f0f0;
        padding: 2px;
        border-radius: 5px;
        text-align: center;
    '>
        <h1 style='color: #333;'>🚀GENAI FOR ADVANCED DOCUMENT PROCESSING</h1>
    </div>
""", unsafe_allow_html=True)

st.markdown("<h2 style='text-align: center; color: #f63366;'> Upload Your File 📂 </h2>", unsafe_allow_html=True)

# File upload
file = st.file_uploader("Choose a file 📑", type=['txt', 'pdf', 'jpg', 'jpeg', 'xlsx','docx'])

if file is not None:
    st.write("File uploaded successfully! 😎")

    if file.type == 'application/pdf':
        pdf_option = st.radio("Select processing method",['OCR','Text'],key='pdf_processing')
        if pdf_option == 'Text':
            pdf_extracted_op = pdf_extractor.extract_pdf(file)
        elif pdf_option == 'OCR':
            pdf_extracted_op = pdf_ocr_extractor.extract_pdf(file)
    if (file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" or file.type == 'application/csv') :
        excel_extracted_op = xlcsv_extractor.extract_xlcsv(file)
    if (file.type.startswith("image/") or file.name.lower().endswith(('.jpg', '.jpeg'))):
        image_extracted_op = image_extractor.extract_image(file)
    elif file.type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
        docx_extracted_op = doc_extractor.extract_doc(file)