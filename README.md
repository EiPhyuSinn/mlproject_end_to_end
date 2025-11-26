#  Insurance Cost Predictor

This project demonstrates an **End-to-End Machine Learning** solution designed to predict **medical insurance charges** based on user-provided health and demographic factors.

<p align="center">
    <img src="https://github.com/user-attachments/assets/3a3e4e5e-c2b2-4b7b-88f7-16cba88ea94f" alt="Insurance Prediction Main UI" width="450" style="margin-right: 15px;">
    <img src="https://github.com/user-attachments/assets/ccb86290-f1ef-4cfe-b9c4-98bc2e35a1e8" alt="Insurance Prediction Result" width="450">
</p>


##  Key Features

* **Modular Pipeline:** Separate steps for streamlined development:
    * Data Ingestion
    * Data Transformation
    * Model Training
* **Robust Preprocessing:** Utilizes `sklearn` Pipelines for automated preprocessing:
    * **Scaling** for numerical features (`StandardScaler`).
    * **One-hot encoding** for categorical features (`OneHotEncoder`).
* **Error Handling:** Implements **Custom Logging** and **Custom Exception** classes for stable operation and easy debugging.
* **UI Deployment:** A simple and neat web interface built with **Flask**.
* **Dataset:** Uses the [Medical Insurance Cost dataset](https://www.kaggle.com/datasets/mosapabdelghany/medical-insurance-cost-dataset).

##  Setup and Run

### 1. Clone the Repository

* To start, clone the project files:
    ```bash
    git clone [https://github.com/YourUsername/mlproject_end_to_end.git](https://github.com/YourUsername/mlproject_end_to_end.git)
    cd mlproject_end_to_end
    ```

### 2. Install Dependencies

* Install all necessary Python libraries (it's recommended to use a virtual environment):
    ```bash
    pip install -r requirements.txt
    ```

### 3. Run the ML Pipeline (Training)

* Execute the script to run the data ingestion, transformation, and model training steps:
    ```bash
    python src/components/data_ingestion.py
    ```
    *(This command kicks off the pipeline, creating the `artifacts/` folder with the preprocessor and trained model.)*

### 4. Run the Web App (Prediction UI)

* Start the Flask server to launch the live prediction interface:
    ```bash
    python app.py
    ```
    *Access the predictor at: **`http://127.0.0.1:5000/`***
http://127.0.0.1:5000/

