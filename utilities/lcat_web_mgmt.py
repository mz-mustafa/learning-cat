import requests
from bs4 import BeautifulSoup
from feature_extraction.lcat_feat_extr_impl import txt_process_spacy
from urllib.parse import urlparse

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


def get_title_desc(html_content):
    soup = BeautifulSoup(html_content, 'lxml')

    # Extract title
    title = None
    if soup.title and soup.title.string:
        title_str = soup.title.string.strip()
        if title_str and title_str not in [None, 'JavaScript is not available.']:
            title = title_str

    # Extract meta description
    meta_desc = None
    if soup.head:  # Check if <head> exists
        meta_desc_tag = soup.head.find('meta', attrs={'name': 'description'})
        if meta_desc_tag:
            # Check for 'content' attribute first, then fallback to 'description' attribute
            meta_desc_content = meta_desc_tag.get('content') or meta_desc_tag.get('description')
            if meta_desc_content and meta_desc_content.strip() != 'JavaScript is not available.':
                meta_desc = meta_desc_content.strip()

    return title, meta_desc


def get_headings(html_content):

    soup = BeautifulSoup(html_content, 'lxml')
    h1_list = [tag.get_text().strip() for tag in soup.find_all('h1') if tag.get_text().strip() and tag.get_text().strip() != 'JavaScript is not available.']
    h2_list = [tag.get_text().strip() for tag in soup.find_all('h2') if tag.get_text().strip() and tag.get_text().strip() != 'JavaScript is not available.']
    h3_list = [tag.get_text().strip() for tag in soup.find_all('h3') if tag.get_text().strip() and tag.get_text().strip() != 'JavaScript is not available.']

    return h1_list, h2_list, h3_list

def get_paras(html_content):
    soup = BeautifulSoup(html_content, 'lxml')
    paras = [tag.get_text().strip() for tag in soup.find_all('p')
             if tag.get_text().strip() and tag.get_text().strip() != 'JavaScript is not available.']
    return paras


def extract_head_props(html_content, keyword):

    h1_best_prac = 0
    h1_list, h2_list, h3_list = get_headings(html_content)

    if h1_list:
        first_h1 = h1_list[0]
        if first_h1:
            if 15 <= len(first_h1) <= 101:
                h1_best_prac = 1

    kw_in_h1 = 1 if is_majority_keyword_present(keyword, ' '.join(h1_list)) else 0
    kw_in_h2 = 1 if is_majority_keyword_present(keyword, ' '.join(h2_list)) else 0
    kw_in_h3 = 1 if is_majority_keyword_present(keyword, ' '.join(h3_list)) else 0

    return h1_list, h2_list, h3_list, h1_best_prac, kw_in_h1, kw_in_h2, kw_in_h3

def is_majority_keyword_present(keyword, content):
    cleansed_keyword = set(txt_process_spacy(keyword, lemmatize=True,is_html=False))
    cleansed_content = set(txt_process_spacy(content, lemmatize=True, is_html=False))
    common_tokens = cleansed_keyword.intersection(cleansed_content)

    return 1 if len(common_tokens) >= len(cleansed_keyword) / 2 else 0

def extract_title_desc_props(html_content, keyword):

    desc_best_prac = 0
    title_best_prac = 0

    title, meta_desc = get_title_desc(html_content)

    if title:
        title_best_prac = 1 if 40 <= len(title) <= 63 else 0

    if meta_desc:
        desc_best_prac = 1 if 100 <= len(meta_desc) <= 163 else 0

    kw_in_title = is_majority_keyword_present(keyword, title)
    kw_in_desc = is_majority_keyword_present(keyword, meta_desc)

    return title, meta_desc, title_best_prac, desc_best_prac, kw_in_title, kw_in_desc



def extract_compre_features(html_content, url):
    soup = BeautifulSoup(html_content, 'html.parser')

    # URL length
    url_len = len(url)
    # Word Count
    words = []
    for tag in soup.find_all(['title', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p']):
        words.extend(tag.get_text().split())
    word_count = len(words)

    # Number of Internal Links
    num_int_links = 0
    parsed_url = urlparse(url)
    for link in soup.find_all('a', href=True):
        if urlparse(link['href']).netloc == parsed_url.netloc:
            num_int_links += 1

    # Number of External Links
    num_ext_links = 0
    for link in soup.find_all('a', href=True):
        if urlparse(link['href']).netloc != parsed_url.netloc:
            num_ext_links += 1

    # Number of Visual Elements
    num_vis_elem = len(soup.find_all(['img', 'video', 'svg', 'canvas']))

    return word_count, num_int_links, num_ext_links, num_vis_elem, url_len
