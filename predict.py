import pandas as pd 
import pickle 
import numpy as np

def predict_insurance():
    # Load the model and preprocessor
    with open('artifacts/model.pkl','rb') as f:
        model = pickle.load(f)
    with open('artifacts/preprocessor.pkl','rb') as f:
        preprocessor = pickle.load(f)

    sample = pd.DataFrame({
        'age': [40],
        'sex': ['1'],          
        'bmi': [25.5],
        'children': [2],
        'smoker': ['0'],
        'region': ['northwest']
    })

    print(f"Sample shape: {sample.shape}")
    print(f"Sample columns: {sample.columns.tolist()}")

    sample_transformed = preprocessor.transform(sample)
    print(f"Transformed sample shape: {sample_transformed.shape}")

    prediction = model.predict(sample_transformed)
    print(f'Prediction: ${prediction[0]:.2f}')

if __name__ == "__main__":
    predict_insurance()