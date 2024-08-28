import toml
import mysql.connector
from mysql.connector import errorcode
import streamlit as st
from mysql.connector import FieldType
from sqlalchemy import create_engine, MetaData, Table, engine


def save_to_db(df, table):
    toml_data = toml.load("secrets.toml")
    db_username = toml_data['mysql']['db_username']
    db_password = toml_data['mysql']['db_password']
    db_host = toml_data['mysql']['db_host']
    db_port = toml_data['mysql']['db_port']
    db_name = toml_data['mysql']['db_name']
    client_flags = toml_data['mysql']['client_flags']
    # ssl_ca = toml_data['mysql']['ssl_ca']

    try:
        conn = mysql.connector.connect(
            user=db_username,
            password=db_password,
            host=db_host,
            port=db_port,
            database=db_name,
            client_flags=client_flags,
            # ssl_ca=ssl_ca
        )


        st.write("Connection established")
    except mysql.connector.Error as err:
        if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
            st.write("Something is wrong with the user name or password")
        elif err.errno == errorcode.ER_BAD_DB_ERROR:
            st.write("Database does not exist")
        else:
            st.write(err)
    else:
        cursor = conn.cursor()

        # Drop previous table of same name if one exists
        cursor.execute(f"DROP TABLE IF EXISTS `{table}`;")
        st.write("Finished dropping table (if existed).")

        # Create a SQLAlchemy engine
        engine = create_engine(f"mysql+mysqlconnector://{db_username}:{db_password}@{db_host}:{db_port}/{db_name}")

        # Write DataFrame to MySQL database
        df.to_sql(table, con=engine, if_exists='append', index=False)

        # Cleanup
        conn.commit()
        cursor.close()
        conn.close()
        print("Done.")
