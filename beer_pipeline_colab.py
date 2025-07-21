import math
import numpy as py
import pandas as pd
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
#from pgmpy.models import BayesianModel
#from pgmpy.estimators import MaximumLikelihoodEstimator


# 2. Data Loading & Preprocessing
# --------------------------------
from sklearn.preprocessing import LabelEncoder

def load_and_clean_data(beers_path, ratings_path, like_threshold=3.5):
    og_df = pd.read_csv(beers_path)
        cleaned_df = og_df.copy().drop("rating")
        cleaned_df["bin_rating"] = 0
        for i  in range(len(og_df)):
            rating = og_df.iloc[i]["rating"]
            if rating >= like_threshold:
                cleaned_df.at[i,"bin_rating"] = 1
            else:
                cleaned_df.at[i, "bin_rating"] = 0
        return cleaned_df

load_and_clean_data
# Example usage:
# df, style_encoder = load_and_clean_data('beers.csv', 'ratings.csv')

# --------------------------------------------------
# Preprocessing demonstration
# --------------------------------------------------
# df.head() shows combined and processed data
try:
    df, style_encoder
except NameError:
    df, style_encoder = load_and_clean_data('beers.csv', 'ratings.csv')
print(df.head())

# 3. Exploratory Analysis Exploratory Analysis
# -----------------------
def explore_variables(df):
    """
    Display info, stats, and basic plots.
    """
    print(df.info())
    print(df.describe())
    df[['abv','ibu','ounces']].hist(bins=30, figsize=(12,4))
    top_styles = df['style'].value_counts().nlargest(10)
    top_styles.plot.bar(figsize=(8,4), title='Top 10 Beer Styles')


# 4. Train/Test Split
# -------------------
def split_data(df, test_size=0.2, random_state=42):
    X = df[['abv','ibu','ounces','style']]
    y = df['Like']
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


# 5. Model Structure & Parameter Learning
# ---------------------------------------
model = BayesianModel([
    ('abv', 'Like'),
    ('ibu', 'Like'),
    ('ounces', 'Like'),
    ('style', 'Like')
])

def learn_parameters(model, df):
    """Fit the BayesianModel using Maximum Likelihood Estimation with Laplace smoothing."""
    model.fit(df, estimator=MaximumLikelihoodEstimator, prior_type='laplace')
    return model


# 6. Training
# -----------
# Example usage:
# df = load_and_clean_data('beers.csv', 'ratings.csv')
# X_train, X_test, y_train, y_test = split_data(df)
# train_df = X_train.copy(); train_df['Like'] = y_train
# trained_model = learn_parameters(model, train_df)


# 7. Evaluation
# -------------
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    return acc, prec, rec, f1, cm


# 8. Future Work
# --------------
# - Compare to scikit-learn's CategoricalNB/GaussianNB
# - Structure learning: Tree-Augmented NB
# - Add contextual features (weather, mood)
# - Build a simple UI in Streamlit

# To download this as a .ipynb file, run the following Python code in a Colab cell:
# ```python
# import nbformat as nbf
#
# # Read this script's text
# with open('/content/beer_pipeline_colab.py') as f:
#     script = f.read().splitlines()
#
# nb = nbf.v4.new_notebook()
# cells = []
# for line in script:
#     if line.startswith('# %% [markdown]'):
#         cells.append(nbf.v4.new_markdown_cell('
'.join(script[script.index(line)+1: script.index(line)+1])))
#     # For brevity, assume script is preformatted into notebook cells
# # Instead, you can manually save this notebook via File > Download .ipynb
# ```
