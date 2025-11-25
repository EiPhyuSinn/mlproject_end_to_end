import pandas as pd 
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from src.logger import get_logger
from src.exception import CustomException
from src.utils import save_object
import numpy as np 

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge,Lasso
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

models = {
    "Linear Regression": LinearRegression(),
    "Lasso": Lasso(),
    "Ridge": Ridge(),
    "K-Neighbors Regressor": KNeighborsRegressor(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest Regressor": RandomForestRegressor(),
    "XGBRegressor": XGBRegressor(), 
    "CatBoosting Regressor": CatBoostRegressor(verbose=False),
    "AdaBoost Regressor": AdaBoostRegressor()
}

params = {
    "Linear Regression": {}, # Linear models usually don't need tuning
    "Lasso": {'alpha': [0.01, 0.1, 1, 10]},
    "Ridge": {'alpha': [0.01, 0.1, 1, 10]},
    "K-Neighbors Regressor": {'n_neighbors': [5, 7, 9, 11]},
    "Decision Tree": {
        'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
        'splitter': ['best', 'random'],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10]
    },
    "Random Forest Regressor": {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5]
    },
    "XGBRegressor": {
        'learning_rate': [0.01, 0.1],
        'n_estimators': [100, 200, 500],
        'max_depth': [3, 5, 7]
    },
    "CatBoosting Regressor": {
        'depth': [6, 8, 10],
        'learning_rate': [0.01, 0.05, 0.1],
        'iterations': [30, 50]
    },
    "AdaBoost Regressor": {
        'learning_rate': [0.01, 0.1, 1],
        'n_estimators': [50, 100, 200]
    }
}


def evaluate(test,pred):
    r2 = r2_score(test,pred)
    mae = mean_absolute_error(test,pred)
    rmse = np.sqrt(mean_squared_error(test,pred))
    return r2,mae,rmse


@dataclass
class ModelTrainerConfig:
    train_pkl_path: str = "artifacts/train.pkl"
    test_pkl_path: str = "artifacts/test.pkl"
    trained_model_path: str = "artifacts/model.pkl"

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def initiate_model_trainer(self) -> str:
        try:
            logger = get_logger()
            logger.info("Starting model training process")

            # Load transformed training and testing data
            train_array = pd.read_pickle(self.config.train_pkl_path)
            test_array = pd.read_pickle(self.config.test_pkl_path)

            # Debug: Check shapes
            logger.info(f"Train array shape: {train_array.shape}")
            logger.info(f"Test array shape: {test_array.shape}")

            X_train = train_array[:,:-1]
            y_train = train_array[:,-1]

            X_test = test_array[:,:-1]
            y_test = test_array[:,-1]

            logger.info(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
            logger.info(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

            best_model_name = None
            best_model_score = -np.inf
            best_model = None

            for model_name, model in models.items():
                logger.info(f"Training {model_name}...")

                if model_name in params:
                    model_params = params[model_name]
                else:
                    model_params = {}

                grid_search = GridSearchCV(estimator=model, param_grid=model_params, cv=5, scoring='r2', verbose=3, n_jobs=-1)
                grid_search.fit(X_train, y_train)

                y_pred = grid_search.predict(X_test)
                r2, mae, rmse = evaluate(y_test, y_pred)

                logger.info(f"{model_name} -- R2: {r2:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")

                if r2 > best_model_score:
                    best_model_score = r2
                    best_model_name = model_name
                    best_model = grid_search

            logger.info(f"Best model found: {best_model_name} with R2 score: {best_model_score:.4f}")

            save_object(self.config.trained_model_path, best_model)
            logger.info(f"Trained model saved at {self.config.trained_model_path}")

            return self.config.trained_model_path

        except Exception as e:
            raise CustomException(f"Model training failed: {e}")