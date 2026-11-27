# ❤️ Heart Disease Prediction Using K-Nearest Neighbors (KNN)

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-KNN-orange)
![Scikit Learn](https://img.shields.io/badge/Scikit--Learn-1.6.1-orange?logo=scikitlearn)
![Status](https://img.shields.io/badge/Project-Completed-success)

## 📌 Project Overview

This project uses the **K-Nearest Neighbors (KNN) Algorithm**, a supervised machine learning classification algorithm, to predict the possibility of heart disease based on different health-related parameters.

The model analyzes patient information such as age, gender, blood pressure, cholesterol level, BMI, smoking habits, diabetes, exercise habits, and other medical indicators to classify whether heart disease is present.

The main objective is to explore how machine learning can be applied to healthcare datasets for educational and predictive analysis.

## 🎯 Project Objectives

- Understand and implement the K-Nearest Neighbors algorithm.
- Analyze heart disease-related health information.
- Build a machine learning classification model.
- Predict heart disease status using patient health parameters.
- Understand the importance of feature selection and preprocessing.
- Save and reuse a trained machine learning model using Pickle.

## 🧠 About the K-Nearest Neighbors Algorithm

K-Nearest Neighbors (KNN) is a supervised machine learning algorithm used for classification and regression problems.

It predicts the class of a new data point by identifying its nearest neighbors in the training dataset.

### How KNN Works

1. Select the number of neighbors (K).
2. Calculate the distance between the new data point and training samples.
3. Identify the K nearest neighbors.
4. Count the classes of the nearest neighbors.
5. Assign the majority class as the final prediction.

### Model Configuration

The supplied trained model contains the following configuration:

| Parameter | Value |
|---|---|
| Algorithm | K-Nearest Neighbors |
| Model Type | KNN Classifier |
| Number of Neighbors | 49 |
| Weights | Uniform |
| Distance Metric | Minkowski |
| Search Algorithm | Auto |
| Number of Input Features | 20 |
| Output Classes | 0 and 1 |

## 📂 Dataset Information

The project uses the `heart_disease(1).csv` dataset.

**Dataset overview:**

- Total records: 10,000
- Total columns: 21
- Input features: 20
- Target column: `Heart_Disease_Status`
- Prediction type: Binary Classification

### Dataset Features

The dataset contains the following 20 input features:

| Feature | Description |
|---|---|
| Age | Patient's age |
| Gender | Patient's gender |
| Blood_Pressure | Blood pressure level |
| Cholesterol_Level | Cholesterol level |
| Exercise_Habits | Regular exercise habits |
| Smoking | Smoking status |
| Family_Heart_Disease | Family history of heart disease |
| Diabetes | Diabetes status |
| BMI | Body Mass Index |
| High_Blood_Pressure | High blood pressure status |
| Low_HDL_Cholesterol | Low HDL cholesterol status |
| High_LDL_Cholesterol | High LDL cholesterol status |
| Alcohol_Consumption | Alcohol consumption level |
| Stress_Level | Patient's stress level |
| Sleep_Hours | Average sleep hours |
| Sugar_Consumption | Sugar consumption level |
| Triglyceride_Level | Triglyceride level |
| Fasting_Blood_Sugar | Fasting blood sugar level |
| CRP_Level | C-reactive protein level |
| Homocysteine_Level | Homocysteine level |

### Target Variable

`Heart_Disease_Status`

- **Yes:** Heart disease is present.
- **No:** Heart disease is not present.

## 🛠️ Technologies Used

- **Python** – Programming language.
- **NumPy** – Numerical operations.
- **Pandas** – Data manipulation and analysis.
- **Scikit-learn** – Machine learning implementation.
- **Pickle** – Saving and loading the trained model.
- **Jupyter Notebook** – Model development and experimentation.

## 📁 Project Structure

```text
Heart-Disease-Prediction/
│
├── heart_disease(1).csv
│
├── KNN_model(1).pkl
│
├── README.md
│
└── requirements.txt
```

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/Heart-Disease-Prediction.git
```

### 2. Navigate to the Project Directory

```bash
cd Heart-Disease-Prediction
```

### 3. Create a Virtual Environment (Optional)

```bash
python -m venv venv
```

Activate the environment:

**Windows:**
```bash
venv\Scripts\activate
```

**Linux / macOS:**
```bash
source venv/bin/activate
```

### 4. Install Required Libraries

```bash
pip install numpy pandas scikit-learn==1.6.1
```

The supplied model was serialized with Scikit-learn 1.6.1. Using the same version is recommended for compatibility.

## 💻 Loading the Trained Model

The trained KNN model is saved in the `KNN_model(1).pkl` file.

Use the following Python code to load the model:

```python
import pickle

# Load the trained KNN model
with open("KNN_model(1).pkl", "rb") as file:
    model = pickle.load(file)

# Display model information
print(model)

# Display the expected input features
print(model.feature_names_in_)

# Display model configuration
print("Number of Neighbors:", model.n_neighbors)
print("Distance Metric:", model.metric)
```

## 🔍 Model Prediction

The trained model expects 20 numerical input features in the following order:

```python
[
    'Age',
    'Gender',
    'Blood_Pressure',
    'Cholesterol_Level',
    'Exercise_Habits',
    'Smoking',
    'Family_Heart_Disease',
    'Diabetes',
    'BMI',
    'High_Blood_Pressure',
    'Low_HDL_Cholesterol',
    'High_LDL_Cholesterol',
    'Alcohol_Consumption',
    'Stress_Level',
    'Sleep_Hours',
    'Sugar_Consumption',
    'Triglyceride_Level',
    'Fasting_Blood_Sugar',
    'CRP_Level',
    'Homocysteine_Level'
]
```

**Important:** The CSV contains categorical values and missing data. The supplied model expects numerical input. The original encoding, missing-value handling, and any scaling steps are not included in the supplied files.

Therefore, new inputs must undergo the exact same preprocessing used during model training before calling `model.predict()`.

## 📊 Model Information

The supplied model is a trained `KNeighborsClassifier`.

- Training samples stored in the model: 5,653.
- Input features: 20.
- Number of neighbors: 49.
- Prediction classes: 0 and 1.

**Note:** The supplied files do not include an evaluation report. Accuracy, precision, recall, and F1-score should be calculated using a separate test dataset rather than assumed.

## ⚠️ Limitations

- KNN performance depends on the quality of the input data.
- Missing values must be handled before prediction.
- Categorical variables require appropriate numerical encoding.
- Feature scaling can significantly affect distance-based predictions.
- KNN can become computationally expensive with larger datasets.
- Predictions may be affected by class imbalance and dataset bias.

## 🚨 Disclaimer

This project is developed for educational and machine learning demonstration purposes only.

It is not intended to provide medical diagnoses, treatment recommendations, or professional healthcare advice.

Always consult a qualified healthcare professional for medical decisions.

## 👨‍💻 Author

**Guniganti Kalyan**

Aspiring Data Scientist | Machine Learning Enthusiast | AI Engineer

GitHub: [Add Your GitHub Profile Link](https://github.com/)

LinkedIn: [Add Your LinkedIn Profile Link](https://linkedin.com/)

---

⭐ If you find this project useful, consider giving the repository a star!

**Thank you for visiting my project!** ❤️
