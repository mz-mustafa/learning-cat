import sqlite3
import pandas as pd
from  sqlalchemy import create_engine
from web_mgmt import fetch_html_content


def connect_db():
    try:
        connection = sqlite3.connect('dataset.db')
        cur = connection.cursor()
        return connection, cur
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None

def create_table(conn, c, q):
    try:
        c.execute(q)
        conn.commit()
        return True
    except Exception as e:
        print(f"An error occurred: {e}")
        return False

def dataframe_insert_db(df, table_name, conn):
    try:
        engine = create_engine('sqlite://', echo=False)
        with conn:
            df.to_sql(table_name, conn, if_exists='append', index=False)
        return True
    except Exception as e:
        print(f"An error occurred: {e}")
        return False

def delete_all_rows(conn, c, table_name):
    try:
        c.execute(f"DELETE FROM {table_name}")
        conn.commit()
        return True
    except Exception as e:
        print(f"An error occurred: {e}")
        return False


def table_exists(conn, table_name):
    c = conn.cursor()
    c.execute('''SELECT count(name) FROM sqlite_master WHERE type='table' AND name=?''', (table_name,))
    # if the count is 1, then table exists
    if c.fetchone()[0] == 1:
        return True
    return False


#CREATE TABLES FOR RAW AND FEATURE DATASETS
def create_raw_dataset_table(conn, c):
    query = '''
    CREATE TABLE raw_dataset (
        keyword TEXT,
        position INTEGER,
        link TEXT,
        snippet_url TEXT,
        status_code INTEGER,
        number_of_redirect INTEGER,
        html_content TEXT
    );
    '''
    success = create_table(conn,c,query)
    return success

def csv_df_db(csv_dataset,table_name,conn):

    df = pd.read_csv(csv_dataset)
    success = dataframe_insert_db(df,table_name,conn)
    return success

def get_empty_html_content_rows(c, table_name, batch_size):
    c.execute(f"SELECT snippet_url FROM {table_name} WHERE html_content IS NULL LIMIT ?", (batch_size,))
    rows = c.fetchall()
    return {url[0]: None for url in rows}  # use url[0] to get the first (and only) item in the tuple


def update_html_content(conn, c, table_name, data):
    try:
        for url, html_content in data.items():
            c.execute(f"UPDATE {table_name} SET html_content = ? WHERE snippet_url = ?", (html_content, url))
        conn.commit()
        return True
    except Exception as e:
        print(f"An error occurred: {e}")
        return False


#TEST TEST FUNCTION FOR RAW DATA IMPORT INTO DB
def test_raw_data_to_DB(raw_table_name, raw_dataset_csv):
    conn, c = connect_db()
    if conn is not None and c is not None:
        s1 = True
        if not table_exists(conn, raw_table_name):
            s1 = create_raw_dataset_table(conn, c)
        if s1:
            print("Table exists or created successfully")
            s2 = csv_df_db(raw_dataset_csv, raw_table_name, conn)
            if s2:
                print("Data inserted")
            else:
                print("Insertion failed")
        else:
            print("Table creation failed")
    conn.close()


#TEST FUNCTION TO DELETE DATA IN THE DB
def test_delete_data_from_table(t_name):
    conn, c = connect_db()
    s1 = delete_all_rows(conn,c,t_name)
    if s1:
        print("Data has been deleted from", t_name)
    else:
        print("Error; data not delete from ",t_name)


#TEST FUNCTION TO UPDATE FETCH HTML FOR URLS AND SAVE THEM BACK TO DB AS A BATCH

# Connect to the database
def test_update_html_content_in_db(table_name,batch_size):
    conn, c = connect_db()

    if conn is not None and c is not None:
        print("DB accessed")
        # Get rows where html_content is empty
        data = get_empty_html_content_rows(c, table_name, batch_size)
        if data is not None:
            print("URLs fetched from DB, now proceeding to HTML fetch from URLs")
        else:
            print("All records have HTML content populated")
        # Close the connection
        conn.close()

        # Fetch HTML content
        data = fetch_html_content(data)

        # Reconnect to the database
        conn, c = connect_db()

        if conn is not None and c is not None:
            # Update the html_content in the database
            success = update_html_content(conn, c, table_name, data)

            if success:
                print("HTML content was updated successfully!")
            else:
                print("HTML content update failed!")

            # Close the connection
            conn.close()
        else:
            print("Failed to connect to the database.")
    else:
        print("Failed to connect to the database.")


#TESTING CODE
table = "raw_dataset"
csv_file = 'Data/AI_RawData_Sample.csv'
#test_raw_data_to_DB(table, csv_file)
#test_delete_data_from_table(table)
#test_update_html_content_in_db(table,489)
