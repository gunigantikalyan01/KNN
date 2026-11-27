# Heart Disease Prediction with K-Nearest Neighbors Algorithm

This project uses a **K-Nearest Neighbors (KNN)** classifier to predict heart disease from patient data. KNN is a simple, **instance-based** learning algorithm: it stores the training examples and assigns a class to a new sample by a **majority vote** of its *k* closest neighbors in feature space. In practice, this means the model finds the *k* training points nearest to an input and returns the most common class among them. Because KNN relies on distance in feature space, input features should be **numeric** and on comparable scales.

The model in this repository is trained and evaluated on the UCI Cleveland Heart Disease dataset. Each record in the data represents a patient and includes various health measurements and a target label indicating presence (1) or absence (0) of heart disease. For example, the dataset features include **age**, **sex** (0=female, 1=male), **cp** (chest pain type), **trestbps** (resting blood pressure), **chol** (serum cholesterol), **fbs** (fasting blood sugar), **restecg** (resting ECG results), **thalach** (max heart rate), **exang** (exercise-induced angina), **oldpeak** (ST depression), **slope** (peak exercise ST segment), **ca** (number of major vessels), **thal** (thalassemia), and the **target** label. In the target column, 0 means *no heart disease* and 1 means *heart disease present*. (Exact column definitions are documented in the source dataset.)

## Dataset

- **Source:** The data (`heart_disease(1).csv`) is the UCI Cleveland heart disease dataset (publicly available on Kaggle). It contains patient health metrics and diagnoses used to train and test the model.
- **Features:** There are 14 columns (features plus label). Key features include:
  - `age`, `sex`, `cp`, `trestbps`, `chol`, `fbs`, `restecg`, `thalach`, `exang`, `oldpeak`, `slope`, `ca`, `thal`
  - **`target`** – heart disease label (0 = absence, 1 = presence).
- **Format:** The file is a CSV with a header row. Each row is one patient. All features are numeric (categorical values are encoded as integers). No missing values are assumed.
- **Usage:** The data file should be placed in the project directory. It is used both for training the KNN model and for any future inference examples.

## Preprocessing & Model

- **Feature Scaling:** Since KNN uses distance metrics, it is important that all features are on similar scales. In training, each feature was likely standardized (zero mean, unit variance). To use the model correctly, new input data should be preprocessed in the same way (e.g. apply the same scaling). Failure to normalize features can hurt performance.
- **Model Artifact:** The trained model is provided as `KNN_model.pkl`. This is a pickled/scikit-learn KNN classifier (likely with a specified `n_neighbors` and trained on the scaled dataset). You can load it in Python with `joblib.load("KNN_model.pkl")` or `pickle.load`.
- **KNN Configuration:** (Not directly verifiable from the file) The chosen number of neighbors *k* is not listed here. In a similar example project, `k=7` was found optimal, but you should verify or tune *k* for your own data.

## Requirements / Setup

1. **Python Version:** Use Python 3.x.
2. **Install Dependencies:** In a virtual environment, install required libraries. For example:
   ```bash
   pip install numpy pandas scikit-learn
   ```
   (Optional: `joblib` is included with scikit-learn; if you use it, no extra install is needed.)
3. **Files:** Ensure the following files are in your working directory:
   - `heart_disease(1).csv` (the dataset)
   - `KNN_model.pkl` (the trained KNN model)
4. **Additional Tools:** If you wish to use a web interface, the original project used [Streamlit](https://streamlit.io/) (see example `requirements.txt`). Otherwise, no GUI is required.

## Usage

After setup, you can load the model and make predictions. For example:

```python
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier  # or import joblib, pickle

# Load the trained KNN model
import joblib
model = joblib.load("KNN_model.pkl")

# Load the dataset (if needed)
df = pd.read_csv("heart_disease(1).csv")

# Prepare input features (example: drop the target column if present)
X = df.drop(columns=["target"], errors="ignore")

# Make predictions on the dataset (or on new samples)
predictions = model.predict(X)  # returns array of 0/1
print(predictions[:5])
```

1. **Training (Optional):** If you want to retrain or evaluate the model: split `heart_disease(1).csv` into train/test sets, standardize the features (e.g. using `StandardScaler`), then fit a `KNeighborsClassifier`. Example (pseudocode):
   ```python
   from sklearn.model_selection import train_test_split
   from sklearn.preprocessing import StandardScaler
   
   X = df.drop("target", axis=1)
   y = df["target"]
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
   scaler = StandardScaler().fit(X_train)
   X_train_scaled = scaler.transform(X_train)
   X_test_scaled = scaler.transform(X_test)
   knn = KNeighborsClassifier(n_neighbors=7)
   knn.fit(X_train_scaled, y_train)
   ```
2. **Prediction:** For a new patient, create a feature array (matching the 13 input features in order) and call `model.predict([features])`. The output will be 0 or 1.

*Note:* Replace placeholder code with the actual script names or function calls as needed in your project. The above illustrates typical usage.

## Evaluation and Metrics

The provided files do not include test-set performance metrics. After training or loading the model, you should evaluate it on a held-out test set to compute accuracy, recall, F1-score, etc. (As an example, a similar KNN model on this data achieved about 91.8% accuracy, but your results may vary.) *Placeholder:* _Compute and report your model’s evaluation metrics here (accuracy, precision, recall, F1)._

## Limitations

- **Data Limitations:** The UCI Cleveland dataset is relatively small (~300 records) and may not represent all patient populations. Model performance can be limited by data size and quality.
- **Feature Sensitivity:** KNN is sensitive to feature scales and irrelevant features. Ensure proper scaling and consider feature selection to improve results.
- **Computational Cost:** KNN requires comparing a query to all training points, which can be slow for large datasets. Prediction time grows with dataset size.
- **No Causal Insight:** KNN makes predictions based purely on distance; it does not provide causal explanations. It assumes similar feature values imply similar outcomes.
- **Imbalanced Classes:** If heart disease cases are much more or less frequent than non-cases, predictions may be biased. Check class distribution and consider techniques like weighting or sampling if needed.

## Responsible Use

- **Not Medical Advice:** This model is for educational/demo purposes only. It is **not** a medical device or a substitute for professional healthcare. A prediction of “heart disease” does not constitute a diagnosis. Always consult qualified medical professionals for health decisions.
- **Data Privacy:** The dataset contains de-identified patient records. If you use real patient data, follow all applicable data privacy and protection regulations (e.g. HIPAA, GDPR).
- **Bias and Fairness:** The model reflects biases in the training data. Ensure you understand the data’s demographic coverage before applying the model widely.
- **Accuracy Caveat:** No machine learning model is 100% accurate. The KNN model may misclassify some cases; use additional tests and clinical judgement.
- **Liability:** Use this code responsibly. The authors disclaim any liability for decisions made based on this model’s output.

---

**References:** The dataset and feature descriptions are from the UCI Cleveland Heart Disease data. KNN algorithm details are discussed in standard literature. These references provide background and context for this implementation.

