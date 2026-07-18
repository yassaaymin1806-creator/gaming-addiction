Gaming Addiction - Model Training (
This notebook covers the data analysis and model training pipeline behind the Gaming Addiction Predictor app. It goes from raw data to a trained, deployable classification model.
# Dataset
gaming_addiction.csv — 250 records with 49 columns describing a person's demographics, gaming habits, psychological indicators, and lifestyle, along with an addiction_binary label (1 = addicted, 0 = not addicted).
# Pipeline Overview
# 1. Data Loading & Inspection
 Load the CSV with pandas.
 Inspect shape, dtypes, missing values, duplicates, and summary statistics (head, info, describe, nunique, isnull().sum(), duplicated().sum()).
# 2. Feature Selection
 Dropped the following columns before training:
 user_id — unique identifier, no predictive value.
 country, gender, income_level — excluded from the feature set.
 addiction_score, addiction_severity, toxic_chat_reports, burnout_probability, subscription_status, churn_probability, behavioral_cluster — excluded as they are derived from or highly correlated with the target, to avoid data leakage.
 # 3. Handling Missing Values
 depression_indicator and gpa_or_performance_score had missing values, filled with the column mean.
# 4. Exploratory Data Analysis (EDA)
 Histograms (sns.histplot) for numerical columns to inspect distributions.
 Count plots (sns.countplot) for categorical columns to inspect category frequencies.
 Correlation of all numerical features against the target (addiction_binary) to identify the strongest predictors.
# 5. Encoding & Scaling
 Categorical columns (occupation, preferred_genre, platform, device_type, rank_tier, relationship_status) encoded with LabelEncoder, fitted on the training split   only and reused on the test split.
 Numerical features scaled with StandardScaler, fitted on the training split only.
# 6. Train/Test Split
 80/20 split via train_test_split with random_state=42 for reproducibility.
# 7. Model Training & Evaluation
 Several classifiers were trained and compared on accuracy, classification report, and confusion matrix:
## Model Training & Evaluation

| Model | Training Accuracy | Testing Accuracy |
|---|---|---|
| Logistic Regression | 0.97 | 0.96 |
| SVM (RBF kernel) | 0.98 | 0.96 |
| Decision Tree | 0.99 | 0.94 |
| Random Forest | 0.99 | 0.92 |
| Neural Network | 0.85 | 0.80 |

Logistic Regression was selected for deployment — it reached the top accuracy (tied with SVM) while being simpler, faster, and easier to serve in production.

#8. Saving Artifacts for Deployment
The following are serialized with joblib so the Streamlit app can load them without retraining:

model.pkl — the trained Logistic Regression model.
scaler.pkl — the fitted StandardScaler.
encoders.pkl — a dict of fitted LabelEncoders, one per categorical column.
feature_columns.pkl — the ordered list of feature columns used at training time.
cat_columns.pkl / num_columns.pkl — categorical and numerical column lists.
# Requirements
pandas
numpy
seaborn
matplotlib
scikit-learn
tensorflow (for the neural network baseline only)
joblib
# Notes
All encoders and the scaler must be fitted only on the training split and reused (never refit) on the test split or on new incoming data, to avoid data leakage.
If the dataset is updated, re-run this notebook end-to-end and replace the .pkl artifacts used by the deployed app.
