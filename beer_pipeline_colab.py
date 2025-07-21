# Beer-Liking Prediction Pipeline (Downloadable .ipynb)

*This notebook follows the Milestone 2 structure: Data Prep, Modeling, Evaluation.*

```python
# %matplotlib inline
```

*This notebook follows the Milestone 2 structure: Data Prep, Modeling, Evaluation.*

```python
# %matplotlib inline
```
 (Colab Script)

# 1. Setup & Imports
# ------------------
# Install pgmpy if needed
# !pip install pgmpy

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from pgmpy.models import BayesianModel
from pgmpy.estimators import MaximumLikelihoodEstimator


# 2. Data Loading & Preprocessing
# --------------------------------
from sklearn.preprocessing import LabelEncoder

def load_and_clean_data(beers_path, ratings_path, like_threshold=3.5):
    """
    Load beer attributes and ratings, merge, drop NAs, binarize ratings, encode categorical features.
    """
    # Load CSVs
    beers = pd.read_csv(beers_path)
    ratings = pd.read_csv(ratings_path)
    # Merge on 'beer_id'
    df = pd.merge(beers, ratings, on='beer_id', how='inner')
    # Drop missing values in key columns
    df = df.dropna(subset=['abv', 'ibu', 'ounces', 'style', 'rating_score']).reset_index(drop=True)
    # Binarize ratings into 'Like' / 'Dislike'
    df['Like'] = (df['rating_score'] >= like_threshold).astype(int)
    # Encode 'style' as numeric
    le = LabelEncoder()
    df['Style_enc'] = le.fit_transform(df['style'])
    return df, le

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
