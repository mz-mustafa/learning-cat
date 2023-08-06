import requests
from bs4 import BeautifulSoup


def fetch_html_content(data):
    for url in data.keys():
        try:
            response = requests.get(url)
            if response.status_code == 200 or response.status_code == 202:
                data[url] = response.text
                print(f"HTML fetched for URL: {url}")
            else:
                print(f"Failed to make a request to URL: {url}")
        except Exception as e:
            print(f"An error occurred: {e}")
    return data


def get_text_content_from_url(snip_url):
    # Fetch the HTML content from the URL
    response = requests.get(snip_url)

    if response.status_code == 200 or response.status_code == 202:
        soup = BeautifulSoup(response.content, 'html.parser')

        text_content = ''

        # Get the content of <title>, <h1>, <h2>, <h3>, <h4>, <h5>, <h6>, <meta>, and <p> tags
        for tag in ['title', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'meta', 'p']:
            elements = soup.find_all(tag)
            for element in elements:
                # For meta tags, we want the content attribute's value
                if tag == 'meta':
                    text_content += ' ' + element.get('content', '')
                else:
                    text_content += ' ' + element.get_text()

        return text_content.strip()  # Remove leading and trailing white spaces

    else:
        return ''
