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

### Phase 1: Project Kick-off & Data Design 
**Goals:**
1. Define the “best-beer-of-the-day” problem.
2. Specify all features, including contextual ones (e.g., ABV, IBU).

**Deliverables:**
- Project Charter: problem statement, scope, success criteria.
- Feature Specification: table of variable name, type, source.
- Data Schema diagram.

---

### Phase 2: Data Collection & Preprocessing 
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

### Phase 3: Model Development and Training 
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

#### Conclusion 

![HEAT MAP FOR BEER RATING](image.png)
Looking at the HMM model I provided, let me explain what it actually accomplishes and its limitations:

##### What the Model Does:

###### 1. **Beer Quality Classification**
- Categorizes beers into 3 hidden states: Low_Quality, Medium_Quality, High_Quality
- Based on features like ABV, IBU, and beer style
- Uses a simple scoring system (not very sophisticated)

###### 2. **Rating Prediction Framework**
- Maps continuous ratings (0-5) to discrete bins: Poor, Fair, Good, Excellent
- Learns probability distributions of ratings given quality states
- Creates emission probabilities P(rating | quality)

###### 3. **Basic Recommendations**
- Filters beers by predicted quality state
- Returns highest-rated beers within that quality category
- Simple content-based filtering approach

##### What It **Doesn't** Really Accomplish:

###### 1. **Not True Sequential Modeling**
- The transition matrix is artificially created using brewery groupings
- Beer ratings don't naturally form sequences like speech or time series
- **This is the biggest flaw** - HMMs are meant for sequential data

###### 2. **Limited Feature Usage**
- Only uses ABV, IBU, and style in a very basic way
- Ignores brewery reputation, beer name, serving size effects
- The quality scoring is overly simplistic

###### 3. **No User Personalization**
- Doesn't learn individual user preferences
- Can't adapt to different taste profiles
- One-size-fits-all recommendations

##### What It Could Be Improved To Do:
Better approach - treat beer tasting as a sequence:
- User's beer journey: Light Lager → Pale Ale → IPA → Stout
- Hidden states: User preference evolution
- Observations: Their ratings of different beer types

Or model brewery quality over time:
- Hidden states: Brewery reputation (declining/stable/improving)
- Observations: Beer ratings from that brewery
- Transitions: How brewery quality changes over time



##### Honest Assessment:
This HMM is more of a **proof-of-concept** that demonstrates:
- How to structure an HMM for a domain problem
- Basic parameter learning from data
- Simple recommendation logic

But it's **not particularly effective** because:
1. Beer recommendation isn't naturally a sequential problem
2. A simple collaborative filtering or content-based approach would work better
3. The model artificially forces sequential structure where none exists

##### Better Alternative:
For the beer recommendation system, it would get better results with:
- **Content-based filtering** using beer features
- **Collaborative filtering** if we had user-beer rating pairs  
- **Hybrid approach** combining multiple signals
- **Classification model** to predict if a user will like a beer

---

### Phase 4: Evaluation & Iteration 
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

### Phase 5: Simple UI & Demo 
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
