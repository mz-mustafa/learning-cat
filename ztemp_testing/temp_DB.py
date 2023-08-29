import psycopg2
import csv

# DB Details here

host = "mlscraper20220428.cenc4vci6iyw.eu-west-1.rds.amazonaws.com"
username = "searchscrpr_usr"
password = "MHtS$44fd11*$d"
dbname = "search_scraper_db"


def db_connect(host, username,password,dbname):
    try:
        connection = psycopg2.connect(
            host=host,
            user=username,
            password=password,
            dbname = dbname
        )
        print("Connected to the PostgreSQL server successfully.")
        return connection
    except psycopg2.Error as e:
        print(f"Error: {e}")

def execute_query_on_db(connection, query):
    cursor = connection.cursor()
    cursor.execute(query)
    results = cursor.fetchall()
    cursor.close()
    return results

def list_databases():
    query = "SELECT datname FROM pg_database;"
    connection = db_connect(host, username, password, dbname)
    databases = execute_query_on_db(connection, query)
    return [db[0] for db in databases]

def list_tables():
    query = "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';"
    connection = db_connect(host, username, password, dbname)
    tables = execute_query_on_db(connection, query)
    return [tb[0] for tb in tables]

def fetch_raw_data(metadata_query, data_query):

    connection = db_connect(host, username, password, dbname)
    metadata = execute_query_on_db(connection, query= metadata_query)
    data = execute_query_on_db(connection, query= data_query)
    return metadata, data

def save_dataset_to_csv(metadata, data, file_name):
    # Extract column names from metadata
    headers = [column[0] for column in metadata]

    # Write the data to the CSV file
    with open(file_name, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)

        # Write header
        writer.writerow(headers)

        # Write data rows
        for row in data:
            writer.writerow(row)

"""
CODE BLOCK FOR TO GET MAIN DATA
metadata_query = "SELECT column_name FROM information_schema.columns WHERE table_schema = 'public' AND table_name = 'ml_model_jobresult';"
data_query = "SELECT * FROM ml_model_jobresult WHERE snippet_url is not null;"
metadata, data = fetch_dataset(metadata_query, data_query)
save_dataset_to_csv(metadata, data, file_name='AI_Dataset.csv')
"""
"""
#CODE BLOCK TO GET JOBS DATA
metadata_query_jobresult = "SELECT column_name FROM information_schema.columns WHERE table_schema = 'public' AND table_name = 'ml_model_jobresult';"
metadata_query_job = "SELECT column_name FROM information_schema.columns WHERE table_schema = 'public' AND table_name = 'ml_model_job';"
metadata_query_kw = "SELECT column_name FROM information_schema.columns WHERE table_schema = 'public' AND table_name = 'serp_keywords';"
metadata_query_loc = "SELECT column_name FROM information_schema.columns WHERE table_schema = 'public' AND table_name = 'serp_locations';"

connection = db_connect(host, username, password, dbname)
print(execute_query_on_db(connection, query= metadata_query_jobresult))
print(execute_query_on_db(connection, query= metadata_query_job))
print(execute_query_on_db(connection, query= metadata_query_kw))
print(execute_query_on_db(connection, query= metadata_query_loc))

data_query = "SELECT * FROM ml_model_job LIMIT 10;"
data = execute_query_on_db(connection, query= data_query)
print("Column names:", metadata)
print("data rows:")
for row in data:
    print(row)
"""

"""
#CODE BLOCK TO GET DB SCHEMA IFNFORMATION

databases = list_databases()
print("Available databases:", databases)
tables = list_tables()
print("Available tables: ", tables)
"""