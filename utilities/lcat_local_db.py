import sqlite3
import pandas as pd
from sqlalchemy import create_engine
from utilities.lcat_web_mgmt import fetch_html_content


def connect_db():
    try:
        connection = sqlite3.connect('../dataset.db')
        cur = connection.cursor()
        return connection, cur
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None

def connect_and_get_keywords():
    conn, c = connect_db()
    df = pd.read_sql_query("SELECT * FROM full_dataset", conn)
    return df, df['keyword'].unique()

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
    CREATE TABLE full_dataset (
    keyword            TEXT,
    position           INT,
    link               TEXT,
    snippet_url        TEXT,
    status_code        INT,
    number_of_redirect INT,
    html_content       TEXT,
    kthm_vector        TEXT,
    ktsm_f1            REAL,
    ktsm_f2            REAL,
    ktsm_f3            REAL,
    ktsm_f4            REAL,
    ktsm_f5            REAL,
    max_h1_sim_bert    REAL,
    avg_h1_sim_bert    REAL,
    max_h2_sim_bert    REAL,
    avg_h2_sim_bert    REAL,
    max_h3_sim_bert    REAL,
    avg_h3_sim_bert    REAL,
    h1_best_prac       integer,
    kw_in_h1           integer,
    kw_in_h2           integer,
    kw_in_h3           integer,
    title_best_prac    integer,
    desc_best_prac     integer,
    kw_in_title        integer,
    kw_in_desc         integer,
    title_tf_sim       REAL,
    desc_tf_sim        REAL,
    h1_tf_sim          REAL,
    h2_tf_sim          REAL,
    h3_tf_sim          REAL,
    para_tf_sim        REAL,
    title_sim_bert     REAL,
    desc_sim_bert      REAL,
    avg_para_sim_bert  REAL,
    max_para_sim_bert  REAL
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
    return {url[0]: None for url in rows}


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
                print("data inserted")
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
        print("data has been deleted from", t_name)
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


"""DATABASE UPDATE FUNCTIONS FOR FEATURES"""


def update_topic_features(keyword, updates, similarities):
    print("Updating database with extracted features")
    conn, c = connect_db()
    combined_updates = [
        (
            kthm,
            float(sim[0]),
            float(sim[1]),
            float(sim[2]),
            float(sim[3]),
            float(sim[4]),
            pos,
            keyword
        )
        for (kthm, pos), sim in zip(updates, similarities)
        ]
    query = """
    UPDATE full_dataset 
    SET kthm_vector = ?, ktsm_f1 = ?, ktsm_f2 = ?, ktsm_f3 = ?, ktsm_f4 = ?, ktsm_f5 = ? 
    WHERE position = ? AND keyword = ?
    """
    c.executemany(query, combined_updates)
    conn.commit()
    conn.close()
    print("Update completed, connection closed.")


def update_content_sim_features(keyword, keyword_data):

    print("Updating database with extracted features")
    conn, c = connect_db()


    query = """
    UPDATE full_dataset 
    SET 
        max_h1_sim_bert = CAST(? AS REAL), avg_h1_sim_bert = CAST(? AS REAL), max_h2_sim_bert = CAST(? AS REAL), 
        avg_h2_sim_bert = CAST(? AS REAL), 
        max_h3_sim_bert = CAST(? AS REAL), avg_h3_sim_bert = CAST(? AS REAL), h1_best_prac = ?, kw_in_h1 = ?, 
        kw_in_h2 = ?, kw_in_h3 = ?, title_best_prac = ?, desc_best_prac = ?, 
        kw_in_title = ?, kw_in_desc = ?, title_sim_bert = CAST(? AS REAL), desc_sim_bert = CAST(? AS REAL), 
        avg_para_sim_bert = CAST(? AS REAL), max_para_sim_bert = CAST(? AS REAL), word_count = ?,
                num_int_links = ?, num_ext_links = ?, num_vis_elem = ?, url_len = ?
    WHERE keyword = ? AND position = ?
    """

    # Prepare the data for the update. Use the position we've stored in serp_features
    data_for_update = [serp_features[:-1] + [keyword, serp_features[-1]] for serp_features in keyword_data]

    # Bulk update
    c.executemany(query, data_for_update)
    conn.commit()
    conn.close()
    print("Update completed, connection closed.")


def update_tf_sim_features(batch_data):
    print("Updating database with extracted features")
    conn, c = connect_db()
    print("Updating database with extracted features")
    update_query = """
    UPDATE full_dataset 
    SET title_tf_sim = :title_tf_sim,
        desc_tf_sim = :desc_tf_sim,
        h1_tf_sim = :h1_tf_sim,
        h2_tf_sim = :h2_tf_sim,
        h3_tf_sim = :h3_tf_sim,
        para_tf_sim = :para_tf_sim
    WHERE keyword = :keyword AND position = :position
    """

    transformed_data = []
    for keyword, position, metrics_dict in batch_data:
        data = {
            'keyword': keyword,
            'position': position,
            'title_tf_sim': metrics_dict.get('title_tf_sim', 0),
            'desc_tf_sim': metrics_dict.get('desc_tf_sim', 0),
            'h1_tf_sim': metrics_dict.get('h1_tf_sim', 0),
            'h2_tf_sim': metrics_dict.get('h2_tf_sim', 0),
            'h3_tf_sim': metrics_dict.get('h3_tf_sim', 0),
            'para_tf_sim': metrics_dict.get('para_tf_sim', 0)
        }
        transformed_data.append(data)

    # Batch update
    c.executemany(update_query, transformed_data)
    conn.commit()
    conn.close()
    print("Update completed, connection closed.")

#TESTING CODE
table = "full_dataset"
#csv_file = '../data/AI_RawData.csv'
#test_raw_data_to_DB(table, csv_file)
#test_delete_data_from_table(table)
test_update_html_content_in_db(table,2918)
