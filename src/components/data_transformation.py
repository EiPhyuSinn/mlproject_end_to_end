import pandas as pd 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from src.logger import get_logger
from src.exception import CustomException
from src.utils import save_object
import numpy as np 

class DataTransformationConfig:
    """Configuration for data transformation."""
    def __init__(self):
        self.transformed_train_path = "artifacts/train.csv"
        self.transformed_test_path = "artifacts/test.csv"   
        self.train_pkl_path = "artifacts/train_set.pkl"
        self.test_pkl_path = "artifacts/test_set.pkl"
        self.preprocessor_path = "artifacts/preprocessor.pkl"

class DataTransformation:
    """Class for data transformation."""
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def initiate_data_transformation(self) -> tuple[str, str]:  
        train_df = pd.read_csv(self.config.transformed_train_path)
        test_df  = pd.read_csv(self.config.transformed_test_path)

        target_col = 'charges'

        # Separate features and target BEFORE transformation
        X_train = train_df.drop(columns=[target_col])
        y_train = train_df[target_col]
        X_test = test_df.drop(columns=[target_col])
        y_test = test_df[target_col]

        # Define features explicitly
        numerical_features = X_train.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = X_train.select_dtypes(include=['object']).columns
        
        logger = get_logger()
        try:
            logger.info("Starting data transformation process")
            logger.info(f"Numerical features: {list(numerical_features)}")
            logger.info(f"Categorical features: {list(categorical_features)}")

            numerical_transformer = Pipeline(steps=[
                ('scaler', StandardScaler())
            ])

            categorical_transformer = Pipeline(steps=[
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ])

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numerical_transformer, numerical_features),
                    ('cat', categorical_transformer, categorical_features)
                ])

            # Transform only the features (X), not the target (y)
            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)

            # Combine transformed features with target for saving
            train_array = np.c_[X_train_transformed, y_train.values]
            test_array = np.c_[X_test_transformed, y_test.values]

            save_object(file_path=self.config.train_pkl_path, obj=train_array)
            logger.info(f"Transformed training set saved to {self.config.train_pkl_path}")
            save_object(file_path=self.config.test_pkl_path, obj=test_array)
            logger.info(f"Transformed testing set saved to {self.config.test_pkl_path}")
            save_object(file_path=self.config.preprocessor_path, obj=preprocessor)
            logger.info(f"Preprocessor saved to {self.config.preprocessor_path}")

            return self.config.train_pkl_path, self.config.test_pkl_path
        except Exception as e:
            logger.error(f"Data transformation failed: {e}")
            raise CustomException(f"Data transformation failed: {e}")