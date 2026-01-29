from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load serialized model
model = joblib.load("model/final_xgboost_model.pkl")

@app.route("/")
def home():
    return "SuperKart Sales Prediction API is running"

@app.route("/v1/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        input_df = pd.DataFrame([data])

        prediction = model.predict(input_df)[0]

        return jsonify({
            "predicted_product_store_sales": round(float(prediction), 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)})

## batch prediction
@app.route("/v1/predict/batch", methods=["POST"])
def predict_batch():
    """
    Batch prediction using uploaded CSV file
    """
    try:
        # Validate file presence
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']

        # Read CSV into DataFrame
        input_data = pd.read_csv(file)

        # Drop Product_Id if present (not used for prediction)
        if 'Product_Id' in input_data.columns:
            input_data = input_data.drop(columns=['Product_Id'])

        # Make predictions
        predictions = model.predict(input_data)

        # Prepare response
        response = {
            "predictions": [
                {
                    "row_id": int(idx),
                    "predicted_product_store_sales": round(float(pred), 2)
                }
                for idx, pred in enumerate(predictions)
            ]
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 400
