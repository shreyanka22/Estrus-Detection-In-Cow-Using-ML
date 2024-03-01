from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from itertools import product
import csv
import random

app = Flask(__name__)

# Load your machine learning model and dataset
data = pd.read_csv('Estrus1.csv')  # Replace with the actual path to your dataset
X = data.drop('Estrus', axis=1)
y = data['Estrus']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=60)

# Initialize the classifier
classifier = SVC(kernel='linear', C=1, probability=True)  # Set probability to True
classifier.fit(X_train, y_train)

# Function to predict estrus using the machine learning model and return confidence
def predict_estrus_with_confidence(input_data):
    confidence = classifier.predict_proba(input_data)[:, 1]  # Probability of class 1
    prediction = classifier.predict(input_data)

    return prediction, confidence

# Rest of the code remains the same
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user input from the form
        restlessness = int(request.form['restlessness'])
        standing_to_be_mounted = int(request.form['standing_to_be_mounted'])
        clear_mucus = int(request.form['clear_mucus'])
        swelling_reddening_of_vulva = int(request.form['swelling_reddening_of_vulva'])
        sniffing_nudging = int(request.form['sniffing_nudging'])
        tail_flagging = int(request.form['tail_flagging'])
        mooing = int(request.form['mooing'])

        # Make a prediction using the machine learning model
        input_data = [[restlessness, standing_to_be_mounted, clear_mucus, swelling_reddening_of_vulva,
                       sniffing_nudging, tail_flagging, mooing]]
        prediction = predict_estrus_with_confidence(input_data)

        # If the return is a string, it means the condition check was met
        if isinstance(prediction, str):
            return jsonify({'message': prediction})

        # Otherwise, return the prediction and confidence in JSON format
        return jsonify({'prediction': int(prediction[0]), 'confidence': float(prediction[1])})

    except Exception as e:
        return jsonify({'error': str(e)})

# ... (Remaining code remains unchanged)
@app.route('/get_accuracy', methods=['GET'])
def get_accuracy():
    try:
        # Calculate overall accuracy score
        y_pred = classifier.predict(X_test)
        acc = accuracy_score(y_test, y_pred)*100

        return jsonify({'accuracy': acc})

    except Exception as e:
        return jsonify({'error': str(e)})

# Route for rendering the HTML page
@app.route('/')
def index():
    return render_template('frontend.html')


if __name__ == '__main__':
    app.run(debug=True)
