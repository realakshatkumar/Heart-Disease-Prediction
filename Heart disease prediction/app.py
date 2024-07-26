from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load model
model = pickle.load(open('Models/rf_classifier.pkl', 'rb'))
scaler = pickle.load(open('Models/scaler.pkl', 'rb'))

# Prediction function
def predict(model, scaler, male, age, currentSmoker, cigsPerDay, BPMeds, prevalentStroke, prevalentHyp, diabetes,
            totChol, sysBP, diaBP, BMI, heartRate, glucose):


    # Prepare features array
    features = np.array([[male , age, currentSmoker, cigsPerDay, BPMeds, prevalentStroke,
                          prevalentHyp, diabetes, totChol, sysBP, diaBP, BMI, heartRate, glucose]])

    # Scale the features
    scaled_features = scaler.transform(features)

    # Predict using the model
    result = model.predict(scaled_features)

    return result[0]

# Routes
@app.route("/")
def index():
    return render_template('index.html')

@app.route('/pred', methods=['POST'])
def predict_route():
    if request.method == 'POST':
        male = int(request.form.get('male'))
        age = int(request.form['age'])
        currentSmoker = request.form['currentSmoker']
        cigsPerDay = float(request.form['cigsPerDay'])
        BPMeds = request.form['BPMeds']
        prevalentStroke = request.form['prevalentStroke']
        prevalentHyp = request.form['prevalentHyp']
        diabetes = request.form['diabetes']
        totChol = float(request.form['totChol'])
        sysBP = float(request.form['sysBP'])
        diaBP = float(request.form['diaBP'])
        BMI = float(request.form['BMI'])
        heartRate = float(request.form['heartRate'])
        glucose = float(request.form['glucose'])

        prediction = predict(model, scaler, male, age, currentSmoker, cigsPerDay, BPMeds, prevalentStroke, prevalentHyp, diabetes, totChol, sysBP, diaBP, BMI, heartRate, glucose)
        prediction_text = "The Patient has Heart Disease" if prediction == 1 else "The Patient has No Heart Disease"

        return render_template('index.html', prediction=prediction_text)

if __name__ == '__main__':
    app.run(debug=True)