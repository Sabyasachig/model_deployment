import joblib
import pandas as pd
from flask import Flask, request, jsonify

# Initialize Flask app with a name
churn_predictor_api = Flask("Customer Churn Predictor")

# Load the trained churn prediction model
model = joblib.load("churn_prediction_model_v1_0.joblib")

# Define a route for the home page
@churn_predictor_api.get('/')
def home():
    return "Welcome to the Customer Churn Prediction API!"

# Define an endpoint to predict churn for a single customer
@churn_predictor_api.post('/v1/customer')
def predict_churn():
    # Get JSON data from the request
    customer_data = request.get_json()

    # Extract relevant customer features from the input data
    sample = {
        'CreditScore': customer_data['CreditScore'],
        'Geography': customer_data['Geography'],
        'Age': customer_data['Age'],
        'Tenure': customer_data['Tenure'],
        'Balance': customer_data['Balance'],
        'NumOfProducts': customer_data['NumOfProducts'],
        'HasCrCard': customer_data['HasCrCard'],
        'IsActiveMember': customer_data['IsActiveMember'],
        'EstimatedSalary': customer_data['EstimatedSalary']
    }

    # Convert the extracted data into a DataFrame
    input_data = pd.DataFrame([sample])

    # Make a churn prediction using the trained model
    prediction = model.predict(input_data).tolist()[0]

    # Map prediction result to a human-readable label
    prediction_label = "churn" if prediction == 1 else "not churn"

    # Return the prediction as a JSON response
    return jsonify({'Prediction': prediction_label})

# Define an endpoint to predict churn for a batch of customers
@churn_predictor_api.post('/v1/customerbatch')
def predict_churn_batch():
    # Get the uploaded CSV file from the request
    file = request.files['file']

    # Read the file into a DataFrame
    input_data = pd.read_csv(file)

    # Make predictions for the batch data and convert raw predictions into a readable format
    predictions = [
        'Churn' if x == 1
        else "Not Churn"
        for x in model.predict(input_data.drop("CustomerId",axis=1)).tolist()
    ]

    cust_id_list = input_data.CustomerId.values.tolist()
    output_dict = dict(zip(cust_id_list, predictions))

    return output_dict

# Run the Flask app in debug mode
if __name__ == '__main__':
    churn_predictor_api.run(debug=True)
