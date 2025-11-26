import pandas as pd
import pickle
from flask import Flask, request, render_template

try:
    with open('artifacts/model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('artifacts/preprocessor.pkl', 'rb') as f:
        preprocessor = pickle.load(f)
except FileNotFoundError:
    print("ERROR: model.pkl or preprocessor.pkl not found in 'artifacts/' directory.")
    model, preprocessor = None, None

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def predict_insurance():
    prediction_result = None
    
    if request.method == 'POST':
        try:
            data = {
                'age': [int(request.form['age'])],
                'sex': [request.form['sex']],
                'bmi': [float(request.form['bmi'])],
                'children': [int(request.form['children'])],
                'smoker': [request.form['smoker']],
                'region': [request.form['region']]
            }

            sample = pd.DataFrame(data)
    
            sample['sex'] = sample['sex'].astype('object')
            sample['smoker'] = sample['smoker'].astype('object')
            
            sample_transformed = preprocessor.transform(sample)
            prediction = model.predict(sample_transformed)[0]
            
            prediction_result = f"Estimated Insurance Charge: ${prediction:.2f}"
            
        except Exception as e:
            prediction_result = f"Error in prediction: {e}"
            print(f"Prediction error: {e}")

    return render_template('index.html', prediction_result=prediction_result)

if __name__ == '__main__':
    app.run(debug=True, port=5000)