from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load("house_price_model.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from the request
    data = request.json
    try:
        # Prepare data for prediction
        features = np.array([[
            data['Bedrooms'],
            data['Bathrooms'],
            data['Area'],
            data['Furnishing'],
            data['Tennants']
        ]])

        # Make a prediction
        prediction = model.predict(features)

        # Return the result
        return jsonify({"Predicted Price": prediction[0]})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
