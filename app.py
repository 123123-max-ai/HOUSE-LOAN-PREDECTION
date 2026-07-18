from flask import Flask, request, jsonify
import pandas as pd
import joblib

# Create Flask app
app = Flask(__name__)

# Load trained model
model = joblib.load("house_price_model.pkl")


@app.route("/")
def home():
    return """
    <h1>House Price Prediction API</h1>

    <h3>API is Running Successfully!</h3>

    <p>Use POST /predict to predict house prices.</p>
    """


@app.route("/predict", methods=["POST"])
def predict():

    try:

        data = request.get_json()

        df = pd.DataFrame([data])

        # Convert categorical columns to dummy variables
        df = pd.get_dummies(df)

        # Expected columns used during training
        expected_columns = [
            'area',
            'bedrooms',
            'bathrooms',
            'stories',
            'parking',
            'mainroad_yes',
            'guestroom_yes',
            'basement_yes',
            'hotwaterheating_yes',
            'airconditioning_yes',
            'prefarea_yes',
            'furnishingstatus_semi-furnished',
            'furnishingstatus_unfurnished'
        ]

        # Add missing columns
        for col in expected_columns:
            if col not in df.columns:
                df[col] = 0

        # Keep same order
        df = df[expected_columns]

        # Prediction
        prediction = model.predict(df)

        return jsonify({
            "Predicted House Price": round(float(prediction[0]), 2)
        })

    except Exception as e:

        return jsonify({
            "Error": str(e)
        })


if __name__ == "__main__":
    app.run(debug=True)