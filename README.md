# Project Title

**Team Members:**
- Alexander Tatoian (PID: A18508705)
- Hyungjun Doh (PID: A16969664)
- Jonny Le (PID: A16873166)
- Vy Dang (PID: A18531908)

**Project Summary**
> Predict whether a user will like a beer based on its attributes using a probabilistic (Bayesian) model.

---

## 1. PEAS / Agent Analysis

- **Performance Measure**: Classification accuracy (and optionally precision, recall, F₁) on a held-out test set.
- **Environment**: Beer dataset with features:
  - `ABV` (alcohol by volume, continuous)
  - `IBU` (bitterness units, continuous)
  - `Ounces` (serving size, continuous)
  - `Style` (beer style, categorical)
  - `Rating score` (user rating, continuous; to be binarized)
- **Actuators**: Output **Like** / **Dislike** label (text/UI); optional ranked list by probability.
- **Sensors**: User inputs (mouse/keyboard) selecting beer features for inference.

---

## 2. Project Roadmap & Milestones

Code development will occur in a Jupyter Notebook (`.ipynb`). Below is a phased plan for Milestone 2 and beyond.

### Phase 1: Project Kick-off & Data Design (1 week)
**Goals:**
1. Define the “best-beer-of-the-day” problem.
2. Specify all features, including contextual ones (e.g., ABV, IBU).

**Deliverables:**
- Project Charter: problem statement, scope, success criteria.
- Feature Specification: table of variable name, type, source.
- Data Schema diagram.

---

### Phase 2: Data Collection & Preprocessing (2 weeks)
**Goals:**
1. Gather and clean all datasets.
2. Engineer contextual features for “today.”

**Tasks:**
- Load and merge beer attributes & ratings; handle missing values.
- Binarize ratings into **Like** (≥3.5) / **Dislike**.
- Discretize continuous features or encode categoricals.
- Fetch contextual data (Such as ABV, IBU, and Rating).
- - ABV and IBU will be different to each user since everyone have a different taste to match the user's preference.
- - But for Rating, it is a combination of multiple opinions on whether a beer is worth a try, the Agent will base on the Rating to rank the list of beers to try that matches user's preference.
- Implement preprocessing pipeline (`load_data()`, `clean_data()`, `feature_engineer()`).
![alt text](HMM.png)

**Deliverables:**
- Cleaned dataset ready for modeling.
- Notebook with exploratory plots & summary.

---

### Phase 3: Model Development and Training (2 weeks)
**Goals:**
1. Implement the Hidden-Markov-Model (HMM) recommendation pipeline.
2. Validate inference of today’s best beer.

**Tasks:**
- Define HMM structure and draw graph: Previous state will affect next state, which will result in different observing outputs. We want to "hide" the learning state, like a BlackBox method, and the users should only care about the oberserving sequences.
- Such as, when learning from the users, it will constantly making changes and adapting to their ABV and IBU, which will affects different kind of beer output. So in the beginning, the output might not be refined, but over time and training, it will be.
- Estimate CPTs with Laplace smoothing: 
```     #"""Fit the BayesianModel using Maximum Likelihood Estimation with Laplace smoothing."""
        model.fit(df, estimator=MaximumLikelihoodEstimator, prior_type='laplace')
```
- Implement `recommend_beer(features, model)`.
- Write unit tests for probability sums and smoothing.
- HMM Training File can be found [here](beer_test.ipynb)

**Deliverables:**
- Notebook/module with model code and docstrings.
- README snippet showing model instantiation & `.fit()`.

---

### Phase 4: Evaluation & Iteration (1 week)
**Goals:**
1. Measure recommendation quality.
2. Identify and address weaknesses.

**Tasks:**
- Compute top-1 accuracy vs. held-out feedback; compare to baselines.
- Plot confusion matrix and probability distributions.
- Analyze errors by style and context.
- Tune smoothing α; try Gaussian NB; explore TAN structures.

**Deliverables:**
- README “Evaluation & Results” with metrics, plots, interpretations.
- List of ≥3 concrete improvements.

---

### Phase 5: Simple UI & Demo (1 week)
**Goals:**
1. Build minimal interface (CLI or Streamlit).

**Tasks:**
- Input form for context & mood.
- Display top-3 beers with probabilities.
- Collect feedback for retraining.

**Deliverables:**
- Working demo app with instructions.
- README usage guide and screenshots.

---

## 3. Next Steps
1. Create the Jupyter Notebook and implement Phase 1.
2. Populate README as phases complete.
3. Schedule weekly standups and code reviews.

---

## 4. References
- **pgmpy**: https://pgmpy.org/
- **pandas**: https://pandas.pydata.org/
- **scikit-learn**: https://scikit-learn.org/

*(End of README)*
