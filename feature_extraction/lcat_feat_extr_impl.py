import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np
import string
from gensim import corpora
from gensim.models import LdaModel
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from bs4 import BeautifulSoup
import tensorflow_hub as hub
from transformers import DistilBertTokenizer, DistilBertModel
import torch
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
nlp = spacy.load("en_core_web_lg", disable=["ner", "tagger"])
exclude = set(string.punctuation)

"""
REGION FOR UTILITY FUNCTIONS
"""

def txt_process_spacy(content, lemmatize, is_html):

    # Extract text from HTML if the content is HTML
    if is_html:
        content = BeautifulSoup(content, 'html.parser').get_text()

    # Check if content is None or blank
    if content is None or content.strip() == "":
        return []

    # Tokenize and process the content using spaCy
    doc = nlp(content)

    if lemmatize:
        cleansed_words = [token.lemma_.lower() for token in doc if
                          token.lemma_ not in STOP_WORDS and token.lemma_.isalpha()]
    else:
        cleansed_words = [token.lower_ for token in doc if
                          token.lower_ not in STOP_WORDS and token.is_alpha]

    return cleansed_words


def clean_embedding_list(embedding_list):

    return [embed for embed in embedding_list if embed is not None]

module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/4"
use_model = hub.load(module_url)

"""
REGION FOR KEYWORD TOPIC HYBRID APPROACH
Hypotheses: 
A combination of keyword semantic meaning and topic distribution of a webpage 
will have predictive power over SERP position
"""

#Implementation
#1. create word embedding from keywords (USE used for word embeddings)

def embed_text_use(text):

    if text is None or text.strip() == "":
        return None

    embeddings = use_model([text])
    # Access the tensor, convert it to numpy, and reshape it to 1D array
    return embeddings['outputs'][0].numpy()

#2. Use LDA to generate topic distribution of all SERPs (by industry when this dimension is available)
def perform_lda(clean_corpus, num_topics, num_words):
    dictionary = corpora.Dictionary(clean_corpus)
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in clean_corpus]
    lda_model = LdaModel(doc_term_matrix, num_topics, random_state = 100, id2word = dictionary, passes=10)
    topics = lda_model.print_topics(num_topics,num_words)
    return topics, doc_term_matrix, lda_model

#This function determines the dominant topic for each document and returns
# a dataframe that lists the dom topic no., doc no. perc contrib. and text.
def format_topics_sentences(lda_model, corpus, texts):
    sent_topics_df = pd.DataFrame(columns=['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords', 'Text'])

    for i, row_list in enumerate(lda_model[corpus]):
        row = row_list[0] if lda_model.per_word_topics else row_list
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:
                wp = lda_model.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df.loc[len(sent_topics_df)] = [int(topic_num), round(prop_topic,2),
                                                           topic_keywords, texts[i]]
            else:
                break
    return sent_topics_df

# document-topic distribution for each SERP related to the keyword
def get_doc_topic_dist(doc_term_matrix, lda_model, num_topics):
    doc_topic_vectors = []
    for doc_bow in doc_term_matrix:
        vector = lda_model[doc_bow]  # This gives topic distribution for the doc
        # Convert topic distribution to a dense vector of length equal to number of topics
        dense_vector = [0] * num_topics
        for topic_id, topic_weight in vector:
            dense_vector[topic_id] = topic_weight
        doc_topic_vectors.append(dense_vector)
    return doc_topic_vectors

#3. Combine KW word embedding and SERP topic dist - i.e. single vector (can use several ways to combine)
def collect_combined_vector(updates, combined_vector, position):
    # Convert combined_vector to string format for SQLite storage
    str_vector = ",".join(map(str, combined_vector))
    updates.append((str_vector, position))

def concatenate_vectors(keyword_vector, doc_topic_vector):

    return np.concatenate((keyword_vector, doc_topic_vector))

def process_docs_for_keyword(keyword_vector, doc_topic_vectors, df_keyword):

    keyword_updates = []

    for doc_topic_vector, row in zip(doc_topic_vectors, df_keyword.iterrows()):
        combined_vector = concatenate_vectors(keyword_vector, doc_topic_vector)
        position = row[1]['position']  # Get the position for the current SERP from the dataframe
        collect_combined_vector(keyword_updates, combined_vector, position)

    return keyword_updates
#4. Add to featured engineered DS

#Additional

#Function that extracts the dominant topic for each doc in the corpus
def get_dom_topics(lda_model, doc_term_matrix,corpus,df_keyword):

    df_topic_sents_keywords = format_topics_sentences(lda_model, doc_term_matrix, corpus)

    df_dominant_topic = df_topic_sents_keywords.reset_index()
    df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']

    # Add SERP position to df_dominant_topic
    df_dominant_topic['SERP_Position'] = df_keyword['position'].values

    # Subset the DataFrame to exclude 'Text' column
    df_dominant_topic = df_dominant_topic[['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'SERP_Position']]

    pd.set_option('display.max_columns', None)
    print(df_dominant_topic)

    distribution = df_dominant_topic.groupby('Dominant_Topic')['SERP_Position'].value_counts()

    # Convert the distribution to a DataFrame and reset the index for easier viewing
    distribution_df = distribution.to_frame(name='Count').reset_index()

    return df_topic_sents_keywords, distribution_df

"""
REGION FOR KEYWORD TOPIC COSINE SIMILARITY (with USE word embeddings)
Hypotheses: 
Degree of Semantic Similarity between the keyword and topics of a webpage will have predictive power over SERP position

Approach
1. create USE word embedding from keywords
2. create USE word embedding from fixed number of topics 
3. compute cosine similarity for each keyword and topic  
4. Add functions to calc mean and other measures of above scores. Or we could using them as individual features as well.
"""
#Implementation

#1. create USE word embedding from keywords
# already written above.

#2. create USE word embedding from fixed number of topics

def embed_topics(lda_model, num_topics):
    # This function will get the top words from each topic and embed them
    topic_embeddings = []

    for i in range(num_topics):
        # Get the top words for the topic
        topic_terms = lda_model.show_topic(i, topn=3)
        topic_words = ' '.join([word for word, _ in topic_terms])

        # Embed the topic words
        topic_embedding = embed_text_use(topic_words)
        topic_embeddings.append(topic_embedding)

    return topic_embeddings

#3. compute cosine similarity for each keyword and topic
def compute_similarity_features(keyword_vector, doc_topic_vectors, topic_embeddings):
    # This function computes the weighted cosine similarity between keyword and topic embeddings for each SERP's doc_topic_vectors

    topic_similarity_features = []

    for doc_topic_vector in doc_topic_vectors:
        keyword_to_topic_similarities = []

        for i, topic_embedding in enumerate(topic_embeddings):
            similarity = cosine_similarity(keyword_vector.reshape(1, -1), topic_embedding.reshape(1, -1))
            weighted_similarity = similarity[0][0] * doc_topic_vector[i] # Weighting by the topic's contribution to the SERP
            keyword_to_topic_similarities.append(weighted_similarity)

        topic_similarity_features.append(keyword_to_topic_similarities)

    return topic_similarity_features


"""
REGION FOR KEYWORD CONTENT COSINE SIMILARITY (with USE word embeddings)
Hypotheses: 
Degree of Semantic Similarity between key parts of a webpage and the keyword will have predictive power over SERP position

Approach
1. create word embedding from keywords
2. create word embedding from page content key elements e.g. title, h1, h2, h3
3. compute cosine similarity for embedding of each element and keyword
"""
#CODE HERE
#1 and 2
# Load pre-trained DistilBERT tokenizer and model
distil_bert_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
distil_bert_model = DistilBertModel.from_pretrained('distilbert-base-uncased')

def embed_text_distilbert(text):

    # Check if text is None or blank
    if text is None or text.strip() == "":
        return None
    # Tokenize input text and obtain output from DistilBERT
    inputs = distil_bert_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        output = distil_bert_model(**inputs)

    # Take the average of the token embeddings as the text embedding
    embedding = output.last_hidden_state.mean(dim=1)

    return embedding

#3.
def calculate_similarity(embed_text_list, embed_text):

    if not embed_text_list or embed_text is None or (len(embed_text_list) == 1 and embed_text_list[0] is None):
        return 0, 0

    similarities = [cosine_similarity(embed_list_item.reshape(1, -1), embed_text.reshape(1, -1))[0][0] for embed_list_item in embed_text_list]
    return float(max(similarities)), float(sum(similarities) / len(similarities))


"""
REGION FOR KEYWORD CONTENT TF-IDF AND COSINE SIMILARITY (with BERT word embeddings, weighted thru TD IDF) 
Hypotheses: 
1. The degree of semantic relevance between a search keyword and the content of a webpage, 
when assessed through an algorithm that accounts for both term frequency and context-based 
semantic similarity, is predictive of the webpage's search engine results page (SERP) position.

Approach
1. create BERT word embedding for web page content
2. calculate TF-IDF for each word in the web page content. consider corpur as 50 SERP pages for that keyword. (maybe by industry later) 
3. Get weighted word embeddings. WWE = TDIDF score x WE
4. Aggregate into single vector representing the web page
5. create BERT word embedding for keyword.
6. find cosine similarity between 4 and 5
7. Add to featured engineered DS

"""
#CODE HERE

# 2. Compute term frequencies for the expanded vocabulary in a given content
def compute_term_frequencies(content, expanded_vocab):
    content_tokens = txt_process_spacy(content, lemmatize=False, is_html=False)
    return {word: content_tokens.count(word) for word in expanded_vocab}

# 3. Compute semantic similarity between keyword and content using word2vec

#We can use the semantic similarity already determined using BERT vectors.

# 4. Combine term frequencies and semantic similarity
def compute_combined_metric(tf_dict, similarity_score, weights):
    avg_value = sum(tf_dict.values()) / len(tf_dict) if tf_dict else 0
    return avg_value * weights[0] + similarity_score * weights[1]
def expand_vocab_using_spacy(keyword):

    vectors = nlp.vocab.vectors
    lemmatized_tokens = txt_process_spacy(keyword, lemmatize=True, is_html=False)
    expanded_vocab = set(lemmatized_tokens)  # Initialize with original lemmatized tokens

    for token in lemmatized_tokens:
        token_obj = nlp.vocab[token]
        if token_obj.has_vector:  # Check for vector representation
            vector = token_obj.vector.reshape(1, -1)  # Reshape to 2D array
            most_similar = vectors.most_similar(vector, n=5)

            expanded_vocab.update([nlp.vocab.strings[msi] for msi in most_similar[0][0]])

    return expanded_vocab



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