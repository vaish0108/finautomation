import base64
from openai import OpenAI
import json
import pandas as pd
import streamlit as st
from datetime import datetime
from docx import Document
import db_connect


def trim_string(string, length):
    if len(string) <= length:
        return string
    else:
        return string[:length] + "..."

def remove_code_blocks(text):
    if text.startswith("```python"):
        text = text[len("```python"):]
    if text.startswith("```json"):
        text = text[len("```json"):]
    if text.endswith("```"):
        text = text[:-len("```")]
    return text

def extract_text_per_page(file_path):

    # Read the Word document
    doc = Document(file_path)

    # st.write(doc)

    # Initialize variables
    pages = []
    current_page_text = ""

    # Iterate through paragraphs
    for paragraph in doc.paragraphs:
        # Append the text of the current paragraph to the current page
        current_page_text += paragraph.text + "\n"

        # Check if the paragraph contains a page break
        if paragraph.runs:
            if any(run.text == '\x0c' for run in paragraph.runs):
                # If a page break is detected, add the current page to the list of pages
                pages.append(current_page_text.strip())
                # Reset the current page text
                current_page_text = ""

    # Append the last page (if any)
    if current_page_text:
        pages.append(current_page_text.strip())

    return pages

def extract_doc(file):

    # Select LLM model
    my_ai_model = "gpt-3.5-turbo"

    client = OpenAI(
        # defaults to os.environ.get("OPENAI_API_KEY")
        api_key="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",
    )

    file_path = file.name


  #   # Setup env variables
  #   os.environ['AZURE_OPENAI_API_KEY'] = "b89c14b8480645e4baa4425f6aab541e"
  #   os.environ['AZURE_OPENAI_ENDPOINT'] = "https://open-ai-capstone-project.openai.azure.com/"
  #
  #   client = AzureOpenAI(
  #       api_key=os.getenv("AZURE_OPENAI_API_KEY"),
  #       api_version="2024-02-01",
  #       azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
  #   )
  #
  #   deployment_name='capstone_project_isb'
  #
  #   messages = [
  #           {
  #               "role": "system",
  #               "content": """You are a PDF table extractor, a backend processor.
  # - User input is messy raw text extracted from a PDF page by PyPDF2.
  # - Do not output any body text, we are only interested in tables.
  # - The goal is to identify tabular data, and reproduce it cleanly as comma-separated table.
  # - Preface each output table with a line giving title and 10 word summary.
  # - It is crucial to format the response in the form of a python dict of dict with table title as the main key, followed by "summary", "table number" and "table" as sub keys for each table.
  # - The table should not be a dict but a list of lists.
  # - All the elements of the sub list should be of same length within a list.
  # - The output should be a python dict starting with a '{' and ending with '}'. And all dictionary keys need to be in double quotes only.
  # - Reproduce each separate table found in page."""
  #           },
  #           {
  #               "role": "user",
  #               "content": "raw pdf text; extract and format tables:" + pdf_text_extract
  #           }
  #       ]
  #
  #   api_params = {"model": deployment_name, "messages": messages, "temperature":0}


    # Iterate through each page and extract text
    # page = pdf_reader.pages[0]
    page = extract_text_per_page(file_path)

    for page_text in page:
        # st.write(page_text)
    # page_text = page.extract_text()

        messages = [
            {
                "role": "system",
                "content": """You are a PDF table extractor, a backend processor.
  - User input is messy raw text extracted from a PDF page by PyPDF2.
  - Do not output any body text, we are only interested in tables.
  - The goal is to identify tabular data, and reproduce it cleanly as comma-separated table.
  - Preface each output table with a line giving title and 10 word summary.
  - It is crucial to format the response in the form of a python dict of dict with table title as the main key, followed by "summary", "table number" and "table" as sub keys for each table.
  - The table should not be a dict but a list of lists.
  - All the elements of the sub list should be of same length within a list.
  - The output should be a python dict starting with a '{' and ending with '}'. And all dictionary keys need to be in double quotes only.
  - Reproduce each separate table found in page."""
            },
            {
                "role": "user",
                "content": "raw pdf text; extract and format tables:" + page_text
            }
        ]

        api_params = {"model": my_ai_model, "messages": messages, "temperature":0}
        api_response = client.chat.completions.create(**api_params)
        response_message = api_response.choices[0].message.content

        cleaned_string = remove_code_blocks(response_message)

        # Convert string to dictionary of dictionaries
        dict_of_tables = json.loads(cleaned_string)

        llm_dfs = {}
        # for table in dict_of_tables:
        for j, table in enumerate(dict_of_tables):

            if dict_of_tables[table]['table'] and dict_of_tables[table]['table'][0]:
                table_name = "Table_" + str(dict_of_tables[table]["table number"])

                # Extract header and data
                header = dict_of_tables[table]['table'][0]
                table_data = dict_of_tables[table]['table'][1:]
                header_len = max(len(sublist) for sublist in table_data)
                header_list=[]



                if len(header)!=header_len:
                    st.write ("There is a mismatch between number of columns and number of headers. Please give the headers manually")
                    for k in range(header_len):
                        header  = st.text_input(f"column{k} header: ", key=f"header{k,j}")
                        header_list.append(header)


                else:
                    option = st.radio("Do you want to manually define the headers?", ["Yes", "No"], key=f"manual_header_option{j}")
                    if option == "Yes":
                        header_list = []
                        for k in range(header_len):
                            header  = st.text_input(f"column{k} header: ", key=f"header{k,j}")
                            header_list.append(header)

                    else:
                        header_list = header

                # Provide default headers if no headers are provided
                if not any(header_list):
                    header_list = [f"header_{i+1}" for i in range(len(header_list))]

                # Convert to Pandas DataFrame

                df = pd.DataFrame(table_data, columns=header_list)
                df.fillna('', inplace=True)
                llm_dfs[table_name] = df
                # table = table.replace(' ','')
                file_name = str(file.name)
                # Get table name from user input or use default format
                default_table_name = f'{file_name}_{datetime.now().strftime("%Y%m%d%H%M%S")}'

                table_name_input = st.text_input(f"Enter table name: (Optional)",key =f"table_names{j}")
                if (table_name_input ==''):
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

                if st.button('Download CSV 📥',f'table_button{j}_table_names{j}'):
                    href = download_csv(df)
                    st.markdown(f'<a href="{href}" download="data.csv">Click here to download</a>', unsafe_allow_html=True)


                if st.button("Save to DB  ☁",key =f'table{j}_table_names{j}'):
                    db_connect.save_to_db(df, selected_heading)