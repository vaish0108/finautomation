import base64
from openai import OpenAI
import PyPDF2
import json
import pandas as pd
from spire.pdf import *
import streamlit as st
import db_connect
from datetime import datetime



def remove_code_blocks(text):
    if text.startswith("```python"):
        text = text[len("```python"):]
    if text.startswith("```json"):
        text = text[len("```json"):]
    if text.endswith("```"):
        text = text[:-len("```")]
    return text


def trim_string(string, length):
    if len(string) <= length:
        return string
    else:
        return string[:length] + "..."


def extract_table(pdf_reader, file):
    # Select LLM model
    my_ai_model = "gpt-3.5-turbo"

    client = OpenAI(
        # defaults to os.environ.get("OPENAI_API_KEY")
        api_key="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",
    )

    # # Setup env variables
    # os.environ['AZURE_OPENAI_API_KEY'] = "b89c14b8480645e4baa4425f6aab541e"
    # os.environ['AZURE_OPENAI_ENDPOINT'] = "https://open-ai-capstone-project.openai.azure.com/"
    #
    # client = AzureOpenAI(
    #     api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    #     api_version="2024-02-01",
    #     azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    # )
    #
    # deployment_name='capstone_project_isb'
    #

    # def open_ai_table_from_text(pdf_text_extract):
    #     messages = [
    #         {
    #             "role": "system",
    #             "content": """You are a PDF table extractor, a backend processor.
    #       - User input is messy raw text extracted from a PDF page by PyPDF2.
    #       - Do not output any body text, we are only interested in tables.
    #       - The goal is to identify tabular data, and reproduce it cleanly as comma-separated table.
    #       - Preface each output table with a line giving title and 10 word summary.
    #       - It is crucial to format the response in the form of a python dict of dict with table title as the main key, followed by "summary", "table number" and "table" as sub keys for each table. The table should not be a dict but a list of lists.
    #       - Reproduce each separate table found in page."""
    #         },
    #         {
    #             "role": "user",
    #             "content": "raw pdf text; extract and format tables:" + pdf_text_extract
    #         }
    #     ]
    #
    #     # api_params = {"model": deployment_name, "messages": messages, "temperature":0}
    #
    #     api_response = client.chat.completions.create(
    #         model=deployment_name,
    #         messages=messages
    #     )
    #
    #     # api_response = client.completions.create(**api_params)
    #
    #     response_message = api_response.choices[0].message.content
    #     return response_message

    # Iterate through each page and extract text
    page = pdf_reader.pages[0]
    page_text = page.extract_text()

    messages = [
        {
            "role": "system",
            "content": """You are a PDF table extractor, a backend processor.
    - User input is messy raw text extracted from a PDF page by PyPDF2.
    - Do not output any body text, we are only interested in tables.
    - The goal is to identify tabular data, and reproduce it cleanly as comma-separated table.
    - Preface each output table with a line giving title and 10 word summary.
    - It is crucial to format the response in the form of a python dict of dict with table title as the main key, followed by summary and table as sub keys for each table. The table should not be a dict but a list of lists.
    - Reproduce each separate table found in page."""
        },
        {
            "role": "user",
            "content": "raw pdf text; extract and format tables:" + page_text
        }
    ]

    api_params = {"model": my_ai_model, "messages": messages, "temperature": 0}
    api_response = client.chat.completions.create(**api_params)
    response_message = api_response.choices[0].message.content

    # Convert string to dictionary of dictionaries
    dict_of_tables = json.loads(response_message)

    # Generate list of table names
    table_names = [f"table_{i}" for i in range(1, len(dict_of_tables) + 1)]

    # for table in dict_of_tables:
    for j, table in enumerate(dict_of_tables):

        # Extract header and data
        column_list = dict_of_tables[table]['table'][0]

        # Generate unique replacement values based on position
        replacement_values = [f'default_value_{i}' for i in range(1, len(column_list) + 1)]

        for i in range(len(column_list)):
            if column_list[i] is None or column_list[i] == '':
                column_list[i] = replacement_values[i]

        table_data = dict_of_tables[table]['table'][1:]
        header_list = []

        header_len = max(len(sublist) for sublist in table_data)

        if header_len != len(column_list):
            st.write(
                "There is a mismatch between number of columns and number of headers. Please give the headers manually")
            for k in range(header_len):
                header = st.text_input(f"column{k} header: ", key=f"header{k, j}")
                header_list.append(header)


        else:
            option = st.radio("Do you want to manually define the headers?", ["Yes", "No"],
                              key=f"manual_header_option{j}")
            if option == "Yes":
                header_list = []
                for k in range(header_len):
                    header = st.text_input(f"column{k} header: ", key=f"header{k, j}")
                    header_list.append(header)

            else:
                header_list = column_list


        # Provide default headers if no headers are provided
        if not any(header_list):
            header_list = [f"header_{i+1}" for i in range(len(column_list))]



        # Convert to Pandas DataFrame
        df = pd.DataFrame(table_data, columns=header_list)
        table = table.replace(' ', '')
        file_name = str(file.name)

        # Get table name from user input or use default format
        default_table_name = f'{file_name}_{datetime.now().strftime("%Y%m%d%H%M%S")}'

        table_name_input = st.text_input(f"Enter table name: (Optional)", key=f"table_names{j}")
        if (table_name_input == ''):
            selected_heading = default_table_name
        else:
            selected_heading = table_name_input

        # Print the DataFrame
        st.write("Table Name : ", selected_heading)
        st.write(df)

        # Create a download button
        def download_csv(df):
            csv = df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()  # Base64 encoding
            href = f'data:file/csv;base64,{b64}'  # Creating a download link
            return href

        if st.button('Download CSV 📥', key= f'button{j}_table_names{j}'):
            href = download_csv(df)
            st.markdown(f'<a href="{href}" download="data.csv">Click here to download</a>', unsafe_allow_html=True)

        if st.button("Save to DB  ☁", key=f'table{j}_table_names{j}'):
            db_connect.save_to_db(df, selected_heading)


def extract_pdf(file):
    # Create a PdfDocument object
    pdf = PdfDocument()
    st.write(file.name)

    # Load a PDF file
    pdf.LoadFromFile(file.name)

    selected_page = st.text_input(
        "Please enter the page number you want to process", key="page_number"
    )


    if selected_page == '':
        st.write( " ")
    else:
        # Create new PdfDocument objects
        newPdf_1 = PdfDocument()

        selected_page_num = int(selected_page) -1
        selected_page_number_str = str(selected_page_num)

        # Insert select pages into new PDF file
        newPdf_1.InsertPageRange(pdf, selected_page_num, selected_page_num)

        # pdf_file = "./pdf/OMV_28.pdf"
        pdf_file = "./pdf/" + file.name + "_" + selected_page_number_str + ".pdf"

        # Save the resulting files
        newPdf_1.SaveToFile(pdf_file)

        # Close the PdfDocument objects
        pdf.Close()
        newPdf_1.Close()

        pdf_reader = PyPDF2.PdfReader(pdf_file)
        extract_table(pdf_reader, file)
