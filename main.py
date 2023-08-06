import pandas as pd
import numpy as np

from dataset_mgmt import update_dataset_html
from us_feature_discovery import clean, perform_lda, format_topics_sentences, embed_text
from local_db import connect_db

# Connect to the SQLite database
conn, c = connect_db()
df = pd.read_sql_query("SELECT * FROM raw_dataset_no_duplicates", conn)

unique_keywords = df['keyword'].unique()

# Number of keywords to process
num_keywords = 5  # Set your desired number here

for keyword in unique_keywords[:num_keywords]:
    print(f"Keyword: {keyword}")  # Print the keyword
    #IMPLEMENTING TOPIC MODELING
    df_keyword = df[df['keyword'] == keyword] #df_keywords holds all SERP data for the selected kw
    corpus = df_keyword['html_content'].tolist() #corpus is a list of html content for the selected kw
    clean_corpus = [clean(doc) for doc in corpus] #clear_corpus is a clean ver. of corpus
    topics, doc_term_matrix, lda_model = perform_lda(clean_corpus) #perform lda for docs for one kw
    for topic in topics:
        print(topic)

    doc_topic_matrix = np.zeros((len(doc_term_matrix), 5))

    # Populate the matrix
    for doc_id, bow in enumerate(doc_term_matrix):
        topic_probs = lda_model.get_document_topics(bow)
        for topic_id, prob in topic_probs:
            doc_topic_matrix[doc_id, topic_id] = prob

    """
    # Let's print the matrix and check some individual entries:
    print("Document-Topic Matrix:")
    print(doc_topic_matrix)

    # Print the topic distribution for the first document:
    print("\nTopic distribution for the first document:")
    print(doc_topic_matrix[0])

    # Print the probability that the first document pertains to the first topic:
    print("\nProbability that the first document pertains to the first topic:")
    print(doc_topic_matrix[0][0])
    """

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

    print(distribution_df)
"""
    #IMPLEMENTING KEYWORD
    # Use it on a list of keywords
    #keywords = ["rent car", "Fibre", "cannabidiol epilepsy", "child nutrition drink", "financial support for lennox"]
    # You need to process each keyword separately
    embedding = embed_text(keyword)
    print(f"Keyword: {keyword}")
    print(f"Embedding: {embedding}\n")
"""