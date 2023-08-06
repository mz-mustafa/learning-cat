import numpy as np
import csv
import requests
from bs4 import BeautifulSoup
import pandas as pd


def write_to_csv(data, filename='AI_EngineeredFeaturesDataset.csv'):
    # Open the CSV file
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        # Create a CSV writer
        writer = csv.writer(csvfile)

        # Write the header
        writer.writerow(data[0].keys())

        # Write the data rows
        for row in data:
            writer.writerow(row.values())


def extract_features(snip_url, page_url, kw, num_redir, position, soup):
    return True


def read_csv_and_fetch_html(file_name='AI_RawData.csv'):
    i = 1
    all_data = []

    fieldnames = ['position', 'page_url', 'kw','f01','f02','f03','f04','f05'
                            ,'f06','f07','f08','f09','f10','f11','f12']

    with open('AI_EngineeredFeaturesDataset.csv', 'a', newline='', encoding='utf-8') as outfile:
        # Create a CSV writer
        writer = csv.DictWriter(outfile,fieldnames=fieldnames)
        writer.writeheader()

        # Read the CSV file
        with open(file_name, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)

            # Iterate through each row in the CSV
            for row in reader:
                # Get the URL from the snippet_url column
                snip_url = row['snippet_url']
                page_url = row['link']
                kw = row['keyword']
                num_redir = row['number_of_redirect']
                position = row['position']

                # Fetch the HTML content from the URL
                response = requests.get(snip_url)

                # Check if the request was successful (HTTP status code 200)
                if response.status_code == 200:
                    # Load the HTML content into a variable
                    html_content = response.text

                    # Parse the HTML content with BeautifulSoup
                    soup = BeautifulSoup(html_content, 'lxml')

                    features = extract_features(snip_url,page_url,kw,num_redir,position,soup)

                    writer.writerow(features)
                    #all_data.append(features)
                    print('Processed row ', i)
                    i +=1




# Call the function
read_csv_and_fetch_html()