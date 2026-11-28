
from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# --------------------------------
# 1. Load trained KNN model
# --------------------------------

with open("KNN_model.pkl", "rb") as file:
    model = pickle.load(file)


# --------------------------------
# 2. Encoding dictionaries
# --------------------------------

yes_no = {
    "Yes": 1,
    "No": 0
}

levels = {
    "High": 2,
    "Medium": 1,
    "Low": 0
}

gender_map = {
    "Female": 0,
    "Male": 1
}


# --------------------------------
# 3. Model feature order
# --------------------------------

FEATURES = [
    "Age",
    "Gender",
    "Blood_Pressure",
    "Cholesterol_Level",
    "Exercise_Habits",
    "Smoking",
    "Family_Heart_Disease",
    "Diabetes",
    "BMI",
    "High_Blood_Pressure",
    "Low_HDL_Cholesterol",
    "High_LDL_Cholesterol",
    "Alcohol_Consumption",
    "Stress_Level",
    "Sleep_Hours",
    "Sugar_Consumption",
    "Triglyceride_Level",
    "Fasting_Blood_Sugar",
    "CRP_Level",
    "Homocysteine_Level"
]

# Numeric input features
NUMERIC_FEATURES = [
    "Age",
    "Blood_Pressure",
    "Cholesterol_Level",
    "BMI",
    "Sleep_Hours",
    "Triglyceride_Level",
    "Fasting_Blood_Sugar",
    "CRP_Level",
    "Homocysteine_Level"
]

# Yes / No input features
BINARY_FEATURES = [
    "Smoking",
    "Family_Heart_Disease",
    "Diabetes",
    "High_Blood_Pressure",
    "Low_HDL_Cholesterol",
    "High_LDL_Cholesterol"
]

# High / Medium / Low features
LEVEL_FEATURES = [
    "Exercise_Habits",
    "Alcohol_Consumption",
    "Stress_Level",
    "Sugar_Consumption"
]


# --------------------------------
# 4. Home page
# --------------------------------

@app.route("/")
def home():
    return render_template("index.html")


# --------------------------------
# 5. Prediction route
# --------------------------------

@app.route("/predict", methods=["GET","POST"])
def predict():

    try:

        input_data = {}

        # Convert numeric values
        for feature in NUMERIC_FEATURES:

            value = request.form.get(feature, "").strip()

            if not value:
                raise ValueError(
                    f"Please enter {feature.replace('_', ' ')}"
                )

            number = float(value)

            if not np.isfinite(number):
                raise ValueError(f"Invalid value for {feature}")

            input_data[feature] = number

        # Convert Gender
        gender = request.form.get("Gender", "").strip()

        if gender not in gender_map:
            raise ValueError("Please select a valid gender")

        input_data["Gender"] = gender_map[gender]

        # Convert Yes / No
        for feature in BINARY_FEATURES:

            value = request.form.get(feature, "").strip()

            if value not in yes_no:
                raise ValueError(
                    f"Please select a valid value for {feature}"
                )

            input_data[feature] = yes_no[value]

        # Convert High / Medium / Low
        for feature in LEVEL_FEATURES:

            value = request.form.get(feature, "").strip()

            if value not in levels:
                raise ValueError(
                    f"Please select a valid value for {feature}"
                )

            input_data[feature] = levels[value]

        # Create DataFrame in exact model order
        input_df = pd.DataFrame(
            [[input_data[column] for column in FEATURES]],
            columns=FEATURES
        )

        # --------------------------------
        # 6. Send inputs to KNN model
        # --------------------------------

        prediction = int(model.predict(input_df)[0])

        # Convert model output to Yes / No
        result = "Yes" if prediction == 1 else "No"

        # --------------------------------
        # 7. Return result to frontend
        # --------------------------------

        return render_template(
            "index.html",
            prediction=prediction,
            result=result
        )

    except ValueError as error:

        return render_template(
            "index.html",
            error=str(error)
        )

    except Exception as error:

        app.logger.exception("Prediction failed")

        return render_template(
            "index.html",
            error="Unable to generate prediction. Please check the model and input data."
        )


# --------------------------------
# 8. Run Flask application
# --------------------------------

if __name__ == "__main__":
    app.run(debug=True)
