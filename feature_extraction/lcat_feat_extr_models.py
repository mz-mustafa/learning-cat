
from feature_extraction.lcat_feat_extr_impl import txt_process_spacy, perform_lda, \
    embed_text_use, embed_topics, get_doc_topic_dist, process_docs_for_keyword, \
    compute_similarity_features, calculate_similarity, embed_text_distilbert, \
    compute_term_frequencies, compute_combined_metric, \
    expand_vocab_using_spacy, clean_embedding_list
from utilities.lcat_local_db import connect_and_get_keywords, update_topic_features, \
    update_content_sim_features, update_tf_sim_features
from utilities.lcat_web_mgmt import extract_head_props, extract_title_desc_props, \
    get_title_desc, get_headings, get_paras, extract_compre_features


def collect_combined_vector(updates, combined_vector, position):
    # Convert combined_vector to string format for SQLite storage
    str_vector = ",".join(map(str, combined_vector))
    updates.append((str_vector, position))


def implement_kw_topic_features(start, end, num_topics, num_words):

    print(f"Starting KW + LDA Topic FE process for {start} to {end} keywords ")
    df, unique_keywords = connect_and_get_keywords()
    keywords_to_process = unique_keywords[start - 1:end]

    for keyword in keywords_to_process:

        # Print the keyword
        print(f"Now Processing Keyword: {keyword}")

        # df_keywords holds all SERP data for the selected kw
        df_keyword = df[df['keyword'] == keyword]

        print(f"Now cleaning corpus of {len(df_keyword)} SERPs")
        # Get cleaned corpus for the keyword
        corpus = df_keyword['html_content'].tolist()
        clean_corpus = [txt_process_spacy(doc,lemmatize=True, is_html=True) for doc in corpus]
        print("Corpus is clean")
        # Perform LDA on the cleaned corpus
        topics, doc_term_matrix, lda_model = perform_lda(clean_corpus, num_topics, num_words)
        print("LDA completed")
        # Get the keyword vector embedding
        keyword_vector = embed_text_use(keyword)

        # Extract the document-topic distribution for each SERP related to the keyword
        doc_topic_vectors = get_doc_topic_dist(doc_term_matrix, lda_model, num_topics)

        # Process documents and gather updates
        keyword_updates = process_docs_for_keyword(keyword_vector, doc_topic_vectors, df_keyword)

        topic_embeddings = embed_topics(lda_model, num_topics)
        print("KW, Topic embeddings and KW+Topic_Dist vector created.")
        # Compute the similarity features for the current keyword and its SERPs
        similarity_features = compute_similarity_features(keyword_vector, doc_topic_vectors, topic_embeddings)

        # Bulk update the database for the current keyword
        update_topic_features(keyword, keyword_updates, similarity_features)


def implement_keyword_content_sim_features(start, end):

    print(f"Starting KW Sim using BERT FE process for {start} to {end} keywords ")
    # DB connection, get the unique keywords
    df, unique_keywords = connect_and_get_keywords()
    keywords_to_process = unique_keywords[start - 1:end]

    for keyword in keywords_to_process:
        print(f"Now processing Keyword: {keyword}")
        df_keyword = df[df['keyword'] == keyword]

        # BERT embedding for the keyword
        keyword_embedding = embed_text_distilbert(keyword)
        print("KW BERT embedding created")
        keyword_data = []  # This list will hold the features for all SERPs of a keyword
        print(f"Now processing {len(df_keyword)} SERPs for the keyword")
        for _, row in df_keyword.iterrows():

            print(f"Processing SERP {row['link']} at position {row['position']}")
            html_content = row['html_content']
            pos = row['position']
            url = row['link']
            # Extract title, description, and their properties
            title, meta_desc, title_best_prac, desc_best_prac, kw_in_title, kw_in_desc = \
                extract_title_desc_props(html_content, keyword)

            # Extract headers and their properties
            h1_list, h2_list, h3_list, h1_best_prac, kw_in_h1, kw_in_h2, kw_in_h3 = \
                extract_head_props(html_content, keyword)

            #Extract some comprehensiveness properties as well, while you're at it.
            word_count, num_int_links, num_ext_links, num_vis_elem, url_len = extract_compre_features(html_content, url)
            para_list = get_paras(html_content)
            print("Content elements fetched, basic features generated")
            # Calculate similarity for H1
            h1_embeddings = [embed_text_distilbert(text) for text in h1_list]
            h1_embeddings = clean_embedding_list(h1_embeddings)
            max_h1_sim_bert, avg_h1_sim_bert = calculate_similarity(h1_embeddings, keyword_embedding)

            # Calculate similarity for H2
            h2_embeddings = [embed_text_distilbert(text) for text in h2_list]
            h2_embeddings = clean_embedding_list(h2_embeddings)
            max_h2_sim_bert, avg_h2_sim_bert = calculate_similarity(h2_embeddings, keyword_embedding)

            # Calculate similarity for H3
            h3_embeddings = [embed_text_distilbert(text) for text in h3_list]
            h3_embeddings = clean_embedding_list(h3_embeddings)
            max_h3_sim_bert, avg_h3_sim_bert = calculate_similarity(h3_embeddings, keyword_embedding)

            # Calculate similarity for title and description
            title_embedding = embed_text_distilbert(title)
            title_sim_bert, _ = calculate_similarity([title_embedding], keyword_embedding)

            desc_embedding = embed_text_distilbert(meta_desc)
            desc_sim_bert, _ = calculate_similarity([desc_embedding], keyword_embedding)

            para_embeddings = [embed_text_distilbert(text) for text in para_list]
            para_embeddings = clean_embedding_list(para_embeddings)
            max_para_sim_bert, avg_para_sim_bert = calculate_similarity(para_embeddings, keyword_embedding)
            print("All similarity features created.")
            # Storing the calculated features in an inner list
            serp_features = [
                max_h1_sim_bert, avg_h1_sim_bert,
                max_h2_sim_bert, avg_h2_sim_bert,
                max_h3_sim_bert, avg_h3_sim_bert,
                h1_best_prac, kw_in_h1, kw_in_h2, kw_in_h3,
                title_best_prac, desc_best_prac,
                kw_in_title, kw_in_desc, title_sim_bert, desc_sim_bert,
                avg_para_sim_bert, max_para_sim_bert, word_count,
                num_int_links, num_ext_links, num_vis_elem, url_len, pos
            ]

            # Append the serp_features list to the keyword_data list
            keyword_data.append(serp_features)
            print(f"FE completed SERP at position {row['position']}")
        # Once all SERPs for the keyword are processed, pass the keyword_data list to the database update function
        print("All SERPs processed for kw:", keyword)
        update_content_sim_features(keyword, keyword_data)


def implement_sim_freq_features(start, end, weights):

    print(f"Starting KW Sim+Frequency FE process for {start} to {end} keywords ")

    # DB connection, get the unique keywords.
    df, unique_keywords = connect_and_get_keywords()
    keywords_to_process = unique_keywords[start - 1:end]

    for keyword in keywords_to_process:
        print(f" Now processing Keyword: {keyword}")
        df_keyword = df[df['keyword'] == keyword]
        expanded_vocab = expand_vocab_using_spacy(keyword)
        print(f"Vocabulary Expanded Successfully for: {keyword} ")
        batch_data = []

        print(f"Now processing {len(df_keyword)} SERPs for the keyword")
        for _, row in df_keyword.iterrows():

            print(f"Processing SERP {row['link']} at position {row['position']}")
            html_content = row['html_content']
            metrics_dict = {}
            h1_list, h2_list, h3_list = get_headings(html_content)
            h1s = " ".join(h1_list)
            h2s = " ".join(h2_list)
            h3s = " ".join(h3_list)
            title, desc = get_title_desc(html_content)
            para = " ".join(get_paras(html_content))
            print("Content elements fetched")
            elements = {'title': title, 'desc': desc, 'h1': h1s, 'h2': h2s, 'h3': h3s, 'para': para}
            sim_columns = {'title': 'title_sim_bert', 'desc': 'desc_sim_bert', 'h1': 'avg_h1_sim_bert',
                           'h2': 'avg_h2_sim_bert', 'h3': 'avg_h3_sim_bert',
                           'para': 'avg_para_sim_bert'}
            print("Computing Similarities and Frequency")
            for el_name, el_content in elements.items():
                tf_dict = compute_term_frequencies(el_content, expanded_vocab)
                metrics_dict[f"{el_name}_tf_sim"] = compute_combined_metric(tf_dict, row[sim_columns[el_name]], weights)
            batch_data.append((keyword, row['position'], metrics_dict))
            print(f"FE for {keyword} completed")
        print("All SERPs processed for kw:", keyword)
        update_tf_sim_features(batch_data)


# TEST CODE
start_kw = 2
end_kw = 200
#implement_kw_topic_features(start_kw, end_kw, num_topics=5, num_words=10)

implement_keyword_content_sim_features(start_kw, end_kw)

#implement_sim_freq_features(start_kw, end_kw, weights=[0.3, 0.7])
