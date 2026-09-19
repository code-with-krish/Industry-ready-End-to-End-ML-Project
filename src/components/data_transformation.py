
# ============================================================
# DATA TRANSFORMATION
# ============================================================

# Data Transformation means:
# 1. Read train and test data.
# 2. Handle missing values.
# 3. Scale numerical columns.
# 4. Convert categorical columns into numerical format.
# 5. Create a preprocessing pipeline.
# 6. Apply the same preprocessing to train and test data.
# 7. Save the preprocessing object for future prediction.


# ============================================================
# 1. IMPORT REQUIRED LIBRARIES
# ============================================================

import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

# ColumnTransformer is used to apply different preprocessing
# techniques to different types of columns.
from sklearn.compose import ColumnTransformer

# SimpleImputer is used to handle missing values.
from sklearn.impute import SimpleImputer

# Pipeline is used to combine multiple preprocessing steps.
from sklearn.pipeline import Pipeline

# OneHotEncoder converts categorical values into numerical values.
# StandardScaler scales numerical values.
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Custom exception for project-level error handling.
from src.exception import CustomException

# Logging is used to track the preprocessing steps.
from src.logger import logging

# os is used to create file paths.
import os

# save_object is used to save the preprocessing object.
from src.utils import save_object


# ============================================================
# 2. DATA TRANSFORMATION CONFIGURATION
# ============================================================

@dataclass
class DataTransformationConfig:

    # Path where the preprocessing object will be saved.
    preprocessor_obj_file_path = os.path.join(
        'artifacts',
        'preprocessor.pkl'
    )


# ============================================================
# 3. DATA TRANSFORMATION CLASS
# ============================================================

class DataTransformation:

    # --------------------------------------------------------
    # Constructor
    # --------------------------------------------------------

    def __init__(self):

        # Create the configuration object.
        self.data_transformation_config = DataTransformationConfig()


    # ========================================================
    # 4. CREATE DATA TRANSFORMER OBJECT
    # ========================================================

    def get_data_transformer_object(self):

        '''
        This function is responsible for data transformation based on the data
        '''

        try:

            # ------------------------------------------------
            # STEP 1: DEFINE COLUMN TYPES
            # ------------------------------------------------

            # Numerical columns.
            numerical_column = [
                'writing score',
                'reading score'
            ]


            # Categorical columns.
            categorical_column = [
                'gender',
                'race/ethnicity',
                'parental level of education',
                'lunch',
                'test preparation course'
            ]


            # ------------------------------------------------
            # STEP 2: NUMERICAL PIPELINE
            # ------------------------------------------------

            # Fill missing numerical values with median
            # and then scale the values.

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )


            logging.info(f"numerical column: {numerical_column}")


            # ------------------------------------------------
            # STEP 3: CATEGORICAL PIPELINE
            # ------------------------------------------------

            # Fill missing categorical values with the
            # most frequent value.
            #
            # Convert categorical values into numerical values.
            #
            # Then scale the values.

            cat_pipeline = Pipeline(

           steps=[
        ("imputer", SimpleImputer(strategy='most_frequent')),
        ('One_Hot_Encoder', OneHotEncoder()),
        ('scaler', StandardScaler(with_mean=False))
    ]

)


            logging.info(f"categorical column: {categorical_column}")


            # ------------------------------------------------
            # STEP 4: COLUMN TRANSFORMER
            # ------------------------------------------------

            # Apply numerical pipeline to numerical columns
            # and categorical pipeline to categorical columns.

            preprocessor = ColumnTransformer(
                [
                    ('num_pipepine', num_pipeline, numerical_column),
                    ('cat_pipeline', cat_pipeline, categorical_column)
                ]
            )


            # Return the complete preprocessing object.

            return preprocessor


        except Exception as e:

            raise CustomException(e, sys)


    # ========================================================
    # 5. INITIATE DATA TRANSFORMATION
    # ========================================================

    def initiate_data_transformation(self, train_path, test_path):

        try:

            # ------------------------------------------------
            # STEP 1: READ TRAIN AND TEST DATA
            # ------------------------------------------------

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)


            logging.info("read train and test data completed")


            # ------------------------------------------------
            # STEP 2: CREATE PREPROCESSING OBJECT
            # ------------------------------------------------

            logging.info("obtainig Preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()


            # ------------------------------------------------
            # STEP 3: DEFINE TARGET COLUMN
            # ------------------------------------------------

            # Correct target column name from the dataset.

            target_column_name = "math score"


            # Numerical columns.

            numerical_column = [
                'writing score',
                'reading score'
            ]


            # ------------------------------------------------
            # STEP 4: SEPARATE INPUT FEATURES AND TARGET
            # ------------------------------------------------

            input_feature_train_df = train_df.drop(
                columns=[target_column_name],
                axis=1
            )


            target_feature_train_df = train_df[target_column_name]


            # ------------------------------------------------
            # STEP 5: SEPARATE TEST FEATURES AND TARGET
            # ------------------------------------------------

            input_feature_test_df = test_df.drop(
                columns=[target_column_name],
                axis=1
            )


            target_feature_test_df = test_df[target_column_name]


            # ------------------------------------------------
            # STEP 6: APPLY PREPROCESSING
            # ------------------------------------------------

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )


            # Fit and transform training data.

            input_feature_train_arr = preprocessing_obj.fit_transform(
                input_feature_train_df
            )


            # Transform testing data using the same
            # preprocessing learned from training data.

            input_feature_test_arr = preprocessing_obj.transform(
                input_feature_test_df
            )


            # ------------------------------------------------
            # STEP 7: COMBINE FEATURES WITH TARGET
            # ------------------------------------------------

            train_arr = np.c_[
                input_feature_train_arr,
                np.array(target_feature_train_df)
            ]


            test_arr = np.c_[
                input_feature_test_arr,
                np.array(target_feature_test_df)
            ]


            # ------------------------------------------------
            # STEP 8: SAVE PREPROCESSING OBJECT
            # ------------------------------------------------

            logging.info(f"Saving preprocessing objects.")


            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )


            # ------------------------------------------------
            # STEP 9: RETURN RESULTS
            # ------------------------------------------------

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )


        except Exception as e:

            raise CustomException(e, sys)

