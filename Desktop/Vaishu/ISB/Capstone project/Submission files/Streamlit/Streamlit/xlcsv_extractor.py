from datetime import datetime
import pandas as pd
from openai import OpenAI
import streamlit as st
import db_connect

def extract_xlcsv(file):

    # Select LLM model
    my_ai_model = "gpt-3.5-turbo"

    client = OpenAI(
        # defaults to os.environ.get("OPENAI_API_KEY")
        api_key="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",
    )

    file_path = file.name

    # Read the Excel file
    df = pd.read_excel(file_path)
    # Get count of Unnamed Headers
    header_list = df.columns.tolist()

    # Count the occurrences of headers starting with "Unnamed:"
    count_unnamed = sum(header.startswith('Unnamed:') for header in header_list)

    # Check if more than one occurrence
    if count_unnamed > 1:
        # Create a dictionary to store row data
        row_data_dict = {}

        #Specify max number of rows to read, based on max promopt size
        rows_to_read = 15

        # Iterate over the first rows_to_read number of rows of the DataFrame
        for index, row in df.head(rows_to_read).iterrows():
            row_data_dict[0] = df.columns.tolist()  # Store column headers
            row_data_dict[index + 1] = row.tolist()  # Store row data

        # Filter out rows with all NaN values
        filtered_data = {k: v for k, v in row_data_dict.items() if not all(pd.isna(x) for x in v)}

        # Define the prompt/question
        prompt = f"""
      Your task is to identify the key in a Python dictionary that contains values of column headers in the form of a Python list.
      Do not be conversational. I need the key as the answer.
      The Python dictionary is provided below with triple backticks:
      ```{filtered_data}```
      """


        # Get completion from OpenAI Chat
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ]

        api_params = {"model": my_ai_model, "messages": messages, "temperature":0}

        api_response = client.chat.completions.create(**api_params)


        # Extract the predicted header row from the completion
        header_row = int(api_response.choices[0].message.content)

        # Read the Excel file again with the identified header row
        new_df = pd.read_excel(file_path, header=header_row)

        file_name = str(file.name)

        # Get table name from user input or use default format
        default_table_name = f'{file_name}_{datetime.now().strftime("%Y%m%d%H%M%S")}'

        table_name_input = st.text_input(f"Enter table name: (Optional)",key ="excel_table1")
        if (table_name_input ==''):
            selected_heading = default_table_name
        else:
            selected_heading = table_name_input


        if st.button("Save to DB  ☁",key ="saving_excel_table1"):
            db_connect.save_to_db(new_df, selected_heading)

        st.write(new_df.head(20))


    else:
        st.write("Did not process the table 🤢")
