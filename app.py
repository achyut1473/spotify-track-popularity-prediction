#app.py

from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
model = pickle.load(open('model.pkl', 'rb'))

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Get values from form
    features = [float(x) for x in request.form.values()]
    
    # Convert to numpy array
    final_input = np.array([features])
    
    # Predict
    prediction = model.predict(final_input)[0]
    
    return render_template('index.html', prediction_text=f'Predicted Popularity: {round(prediction, 2)}')

if __name__ == "__main__":
    app.run(debug=True)