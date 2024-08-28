import base64
import pytesseract
from img2table.ocr import TesseractOCR
from img2table.document import Image
import streamlit as st
import db_connect
from datetime import datetime




def extract_image(file) :

    # Set the Tesseract OCR executable path
    pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

    # Instantiate the TesseractOCR object
    ocr = TesseractOCR(n_threads=1, lang="eng", psm=11)

    # Instantiation of the image
    img = file.name

    #Instantiation of OCR
    ocr = TesseractOCR(n_threads=1, lang="eng")

    # Instantiation of document, either an image or a PDF
    doc = Image(img)

    # Table extraction
    extracted_tables = doc.extract_tables(ocr=ocr,
                                          implicit_rows=False,
                                          borderless_tables=False,
                                          min_confidence=50)



    # Generate list of table names
    table_names = [f"table_{j}" for j in range(1, len(extracted_tables) + 1)]

    for k,table in enumerate (range (0,len(extracted_tables))):

        # table_name = 'table'+str(i)
        # st.write(table_name)

        file_name = str(file.name)

        # Get table name from user input or use default format
        default_table_name = f'{file_name}_{datetime.now().strftime("%Y%m%d%H%M%S")}'

        table_name_input = st.text_input(f"Enter table name: (Optional)",key =table_names[k])
        if (table_name_input ==''):
            selected_heading = default_table_name
        else:
            selected_heading = table_name_input

        # Print the DataFrame
        st.write("Table Name : ", selected_heading)
        st.write(extracted_tables[table].df)


        # Create a download button
        def download_csv(df):
            csv = df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()  # Base64 encoding
            href = f'data:file/csv;base64,{b64}'  # Creating a download link
            return href

        if st.button('Download CSV 📥', key= f'button{k}_table_names{k}'):
            href = download_csv(extracted_tables[table].df)
            st.markdown(f'<a href="{href}" download="data.csv">Click here to download</a>', unsafe_allow_html=True)


        if st.button("Save to DB  ☁",key =f'table{k}_{table_names[k]}'):
            db_connect.save_to_db(extracted_tables[table].df, selected_heading)

        st.write("---------------------------------------------")



