import csv
import pandas as pd


def update_dataset_html(file_name):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(file_name)

    # Ask user whether to get html of the whole dataset, or only rows in which html content column value is empty
    choice = input("Get HTML for the whole dataset (yes/no)? ")
    if choice.lower() == 'no':
        df = df[df['html_content'].isna()]

    # Iterate through each row in the DataFrame
    for index, row in df.iterrows():
        # Get the URL from the snippet_url column
        snip_url = row['snippet_url']

        # Fetch the HTML content from the URL
        html_text = get_html(snip_url)

        # Write the fetched html content into the DataFrame
        df.loc[index, 'html_content'] = html_text
        print("html updated for index", index)

    # Write the DataFrame back to the CSV file
    # df.to_csv(file_name, index=False)
    return df



def get_html_generate_features(file_name):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(file_name)

    # Ask the user the row number (start from 0) to start reading from
    start_row = int(input("Enter the row number to start from: "))

    # Iterate from the start row to the end of the DataFrame
    for index, row in df.iloc[start_row:].iterrows():
        # Read each row into a variable called raw_inst
        raw_inst = row

        # Extract the features
        feature_set = extract_features(raw_inst)

        # Write the features to a CSV file
        write_to_csv(feature_set, index)

def extract_features(raw_inst):
    # This is just a dummy function for now
    return {'feature_set': True}

def write_to_csv(feature_set, row_num, file_name):
    # Open the CSV file in append mode
    with open(file_name, 'a', newline='', encoding='utf-8') as outfile:
        # Create a CSV writer
        writer = csv.DictWriter(outfile, fieldnames=feature_set.keys())

        # Write the row to the CSV file
        writer.writerow(feature_set)

