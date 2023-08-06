import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
from gensim import corpora
from gensim.models import LdaModel
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
nltk.download('punkt')
from bs4 import BeautifulSoup
import tensorflow_hub as hub
import tensorflow_text  # Not used directly but is needed to execute some ops in the graph


nltk.download('stopwords')
nltk.download('wordnet')

stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()

"""
REGION FOR UTILITY FUNCTIONS
"""
def clean(doc):
    soup = BeautifulSoup(doc, 'html.parser')
    text = soup.get_text()  # extract text
    stop_free = " ".join([i for i in text.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = [lemma.lemmatize(word) for word in punc_free.split()]
    return normalized
"""
REGION FOR TOPIC MODELING FUNCTIONS
"""

def perform_lda(clean_corpus):
    dictionary = corpora.Dictionary(clean_corpus)
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in clean_corpus]
    lda_model = LdaModel(doc_term_matrix, num_topics = 5, random_state = 100, id2word = dictionary,passes=10)
    topics = lda_model.print_topics(num_topics=5,num_words=10)
    return topics, doc_term_matrix, lda_model

def format_topics_sentences(lda_model, corpus, texts):
    sent_topics_df = pd.DataFrame(columns=['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords', 'Text'])

    for i, row_list in enumerate(lda_model[corpus]):
        row = row_list[0] if lda_model.per_word_topics else row_list
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:
                wp = lda_model.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df.loc[len(sent_topics_df)] = [int(topic_num), round(prop_topic,2), topic_keywords, texts[i]]
            else:
                break
    return sent_topics_df


"""
REGION FOR KEYWORD TOPIC HYBRID APPROACH
Hypotheses: 
1. Degree of Similarity between Dominant Topics of a webpage and the keyword will have predictive power over SERP position
"""

#Approach
#1. create word embedding from keywords
module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/4"
use_model = hub.load(module_url)

def embed_text(text):
    embeddings = use_model([text])
    return embeddings



#2. Use LDA to generate topic distribution of all SERPs (by industry when this dimension is available)
#3. Combine KW word embedding and SERP topic dist - i.e. single vector (can use several ways to combine)
#4. Add to featured engineered DS


#CODE HERE


"""
REGION FOR KEYWORD CONTENT COSINE SIMILARITY (with word2vec word embeddings)
Hypotheses: 
1. Degree of Semantic Similarity between key parts of a webpage and the keyword will have predictive power over SERP position

Approach
1. create word embedding from keywords
2. create word embedding from page content key elements 
3. compute cosine similarity for each element and keyword  
4. Add functions to calc mean and other measures of above scores. Or we could using them as individual features as well.
5. Add to featured engineered DS

"""
#CODE HERE



"""
REGION FOR KEYWORD CONTENT TF-IDF AND COSINE SIMILARITY (with BERT word embeddings, weighted thru TD IDF) 
Hypotheses: 
1. The degree of semantic relevance between a search keyword and the content of a webpage, 
when assessed through an algorithm that accounts for both term frequency and context-based 
semantic similarity, is predictive of the webpage's search engine results page (SERP) position.

Approach
1. create BERT word embedding for web page content
2. calculate TD-IDF for each word in the web page content. consider corpur as 50 SERP pages for that keyword. (maybe by industry later) 
3. Get weighted word embeddings. WWE = TDIDF score x WE
4. Aggregate into single vector representing the web page
5. create BERT word embedding for keyword.
6. find cosine similarity between 4 and 5
7. Add to featured engineered DS

"""
#CODE HERE


#This is the function I used in the RAM project.
#It apparently finds semantic similarity between the keyword and the page's key contents like title etc.
#I haven't really tested it yet. (as of 1807)
def tfidf_similarity(keyword, title, meta_desc, h1s, h2s):
    vectorizer = TfidfVectorizer()

    def calculate_similarity(vector1, vectors):
        return [cosine_similarity(vector1, vector)[0, 0] for vector in vectors]

    keyword_vector = vectorizer.fit_transform([keyword])
    title_ss = 0
    if title is not None:
        title_vector = vectorizer.transform([title])
        title_ss = cosine_similarity(keyword_vector, title_vector)[0, 0]

    meta_desc_ss = 0
    if meta_desc is not None:
        meta_description_vector = vectorizer.transform([meta_desc])
        meta_desc_ss = cosine_similarity(keyword_vector, meta_description_vector)[0, 0]

    avg_h1_ss = 0
    if len(h1s) > 0:
        h1_vectors = vectorizer.transform(h1s)
        h1_scores = calculate_similarity(keyword_vector, h1_vectors)
        avg_h1_ss = np.mean(h1_scores) if h1_scores else 0

    avg_h2_ss = 0
    if len(h2s) > 0:
        h2_vectors = vectorizer.transform(h2s)
        h2_scores = calculate_similarity(keyword_vector, h2_vectors)
        avg_h2_ss = np.mean(h2_scores) if h2_scores else 0

    return title_ss, meta_desc_ss, avg_h1_ss, avg_h2_ss