from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
import logging

app = Flask(__name__)

# Load the trained model
model = joblib.load('crime_model.pkl')

# Configure logging
logging.basicConfig(level=logging.INFO)

# Route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Route to predict crime
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    latitude = data['latitude']
    longitude = data['longitude']
    area = data['area']
    
    # Prepare the input data
    input_data = pd.DataFrame({
        'latitude': [latitude],
        'longitude': [longitude],
        'area_residential': [1 if area == 'residential' else 0],
        'area_commercial': [1 if area == 'commercial' else 0],
        'area_industrial': [1 if area == 'industrial' else 0]
    })
    
    # Predict
    prediction = model.predict(input_data)
    
    logging.info(f"Received data: {data}, Prediction: {int(prediction[0])}")
    
    return jsonify({'narcotics_crime': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
