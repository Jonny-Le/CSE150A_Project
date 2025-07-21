from google.colab import files

# 1. Write README.md
readme_content = """# Project Title

**Team Members:**
- Alexander Tatoian (PID: A18508705)
- Hyungjun Doh (PID: A16969664)
- Jonny Le (PID: A16873166)
- Vy Dang (PID: A18531908)

**Project Summary:**
Predict whether a user will like a beer based on its attributes using a probabilistic model.

---

## 1. PEAS / Agent Analysis

- **Performance Measure**
  - Classification accuracy (and optionally precision, recall, F₁) on a held-out test set: how often the model correctly predicts “like” vs. “dislike.”

- **Environment**
  - The full beer dataset, containing features:
    - **ABV** (alcohol by volume, continuous)
    - **IBU** (bitterness units, continuous)
    - **Ounces** (serving size, continuous)
    - **Style** (beer style, categorical)
    - **Rating score** (user rating, continuous, to be binarized)

- **Actuators**
  - Text (or UI) output of the predicted label: “like” / “dislike”
  - (Optional) A ranked list of beers by predicted probability of “like.”

- **Sensors**
  - Mouse/keyboard inputs through which a user selects a beer (its feature values) to get a prediction.

---

## 2. Project Roadmap & Milestones

Below is a phased plan outlining objectives, tasks, and deliverables for Milestone 2 and beyond. Code will be developed in a Jupyter Notebook (`.ipynb`), and the repository will reference this structure in the README.

### Phase 1: Project Kick-off & Data Design (1 week)

**Goals**
- Define the “best-beer-of-the-day” problem precisely.
- Specify all features, including contextual ones (weather, mood, day of week).

**Deliverables**
- **Project Charter**: problem statement, scope, success criteria.
- **Feature Specification**: table listing each variable (name, type, source).
- **Data Schema**: diagrams showing how tables (beer attributes, user history, context) join.

---

### Phase 2: Data Collection & Preprocessing (2 weeks)

**Goals**
- Gather and clean all data sources.
- Engineer contextual features for “today.”

**Tasks**
1. **Beer Attributes**
   - Load the beer dataset; handle missing ABV/IBU; discretize continuous features.
2. **User Preferences**
   - Compile past like/dislike history; binarize ratings into `Like`/`Dislike`.
3. **Contextual Data**
   - Fetch weather API for today’s temperature/humidity.
   - Design a simple mood survey input.
4. **Preprocessing Pipeline**
   - Implement and document functions: `load_data()`, `clean_data()`, `feature_engineer()`.

**Deliverables**
- A cleaned, merged DataFrame ready for modeling.
- Jupyter notebook section with exploratory plots & summary.

---

### Phase 3: Model Development (2 weeks)

**Goals**
- Implement the Naive Bayes ranking pipeline.
- Validate basic inference of today’s best beer.

**Tasks**
1. **Structure Definition**
   - Finalize feature-to-`Like` graph (Bayesian network structure).
2. **Parameter Learning**
   - Code CPT estimation with Laplace smoothing.
   - Optionally compare to scikit-learn’s `CategoricalNB` or `GaussianNB`.
3. **Inference Function**
   - Write `recommend_beer(today_features, model)` returning top-k beers.
4. **Unit Tests**
   - Verify that predicted probabilities sum to 1 and that unseen features are handled via smoothing.

**Deliverables**
- A Python module (or notebook) `beer_recommender.py` / `.ipynb` with clear docstrings.
- Code snippet in README demonstrating model definition & `.fit()` call.

---

### Phase 4: Evaluation & Iteration (1 week)

**Goals**
- Measure recommendation quality.
- Identify and fix major weaknesses.

**Tasks**
1. **Offline Metrics**
   - Compute accuracy of “top-1” match vs. held-out user feedback.
   - Baseline: random selection vs. popularity-based top-1.
2. **Visualizations**
   - Confusion matrix heatmap for `Like`/`Dislike`.
   - Distribution of predicted probabilities.
3. **Analysis**
   - Breakdown errors by beer style and context (e.g., hot vs. cold days).
4. **Improvements**
   - Tune smoothing parameter α; try Gaussian NB for ABV/IBU.
   - Explore Tree-Augmented NB or simple feature interactions.

**Deliverables**
- “Evaluation & Results” section in README with metrics, plots, and interpretations.
- A list of at least three concrete next steps.

---

### Phase 5: Simple UI & Demo (1 week)

**Goals**
- Build a minimal interface to demo “best beer for today.”

**Tasks**
1. **Input Form**
   - Command-line script or Flask/Streamlit app to collect today’s context (weather, mood).
2. **Recommendation Display**
   - Show top-3 beers with their probabilities and key feature highlights.
3. **Feedback Loop**
   - Allow the user to mark “liked”/“disliked” and append feedback for future retraining.

**Deliverables**
- A working demo app (with run instructions).
- Final README updated with “How to run” and demo screenshots if applicable.

---

## 3. Next Steps

1. Create the Jupyter Notebook and implement Phase 1 deliverables.  
2. Populate the `README.md` with sections as you complete each phase.  
3. Schedule weekly check-ins and code reviews.

---

## 4. References

- pgmpy: https://pgmpy.org/  
- pandas documentation: https://pandas.pydata.org/  
- scikit-learn documentation: https://scikit-learn.org/stable/  
"""

with open('README.md', 'w') as f:
    f.write(readme_content)

# 2. Write beer_pipeline_colab.py
script_content = """# Beer-Liking Prediction Pipeline (Colab Script)

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
def load_and_clean_data(beers_path, ratings_path, like_threshold=3.5):
    \"\"\"
    Load beer attributes and ratings, merge, drop NAs, binarize ratings.
    \"\"\"
    beers = pd.read_csv(beers_path)
    ratings = pd.read_csv(ratings_path)
    df = pd.merge(beers, ratings, on='beer_id', how='inner')
    df = df.dropna(subset=['abv', 'ibu', 'ounces', 'style', 'rating_score'])
    df['Like'] = (df['rating_score'] >= like_threshold).astype(int)
    return df


# 3. Exploratory Analysis
# -----------------------
def explore_variables(df):
    \"\"\"
    Display info, stats, and basic plots.
    \"\"\"
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
    \"\"\"Fit the BayesianModel using Maximum Likelihood Estimation with Laplace smoothing.\"\"\"
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
"""

with open('beer_pipeline_colab.py', 'w') as f:
    f.write(script_content)

# 3. Offer download
files.download('README.md')
files.download('beer_pipeline_colab.py')
