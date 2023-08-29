from sklearn.model_selection import train_test_split
from utilities.lcat_local_db import connect_and_get_keywords
from supervised_learning.lcat_regressor_models import RandomForestModel


def load_data_from_db(table_name, excluded_columns):
    # Connect to the database
    df, _ = connect_and_get_keywords()

    # Exclude unwanted columns
    df = df.drop(columns=excluded_columns)

    return df

topic_features = ['kthm_vector','ktsm_f1', 'ktsm_f2', 'ktsm_f3', 'ktsm_f4', 'ktsm_f5']
basic_features = ['h1_best_prac','kw_in_h1','kw_in_h2','kw_in_h3','title_best_prac', 'desc_best_prac',
                  'kw_in_title','kw_in_desc']
cont_sim_features = ['max_h1_sim_bert', 'avg_h1_sim_bert', 'max_h2_sim_bert', 'avg_h2_sim_bert',
                     'max_h3_sim_bert', 'avg_h3_sim_bert', 'title_sim_bert', 'desc_sim_bert', 'avg_para_sim_bert', 'max_para_sim_bert']
tf_sim_features = ['title_tf_sim', 'desc_tf_sim', 'h1_tf_sim', 'h2_tf_sim', 'h3_tf_sim', 'para_tf_sim']
# Load the data
meta_columns = ["keyword", "link", "snippet_url",
                    "status_code", "number_of_redirect", "html_content", "rn", "kthm_vector"]
excl_columns = meta_columns
df = load_data_from_db("table1", excl_columns)

# Separate features and target
X = df.drop(columns=["position"])
y = df["position"]
print(f"Proceeding to train using {len(X.columns)} features")
# Split the data (You can also use the split_data method in the classes)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# RandomForest Model
rf_model = RandomForestModel()
rf_model.split_data(X, y)  # Or you can skip this step and directly assign the split data
rf_model.scale_features()
rf_model.train_model()
rf_model.predict()
rf_model.print_metrics()
rf_model.plot_feature_importance(X.columns)

"""
# XGBoost Model
xgb_model = XGBoostModel()
xgb_model.split_data(X, y)  # Or you can skip this step and directly assign the split data
xgb_model.scale_features()
xgb_model.train_model()
xgb_model.predict()
xgb_model.print_metrics()
"""

best_params_rf, best_score_rf = rf_model.perform_random_search(X, y)
print("Best Parameters for RandomForest:", best_params_rf)
print("Best Score for RandomForest:", best_score_rf)
"""
best_params_xgb, best_score_xgb = xgb_model.perform_random_search(X, y)
print("Best Parameters for XGBoost:", best_params_xgb)
print("Best Score for XGBoost:", best_score_xgb)
"""