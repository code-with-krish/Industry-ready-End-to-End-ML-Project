
# ============================================================
# DATA INGESTION
# ============================================================
# Data Ingestion means:
# 1. Read data from a data source.
# 2. Store the raw data.
# 3. Split the data into train and test datasets.
# 4. Store train and test data for the next pipeline steps.
#
# Data can come from different sources such as:
# - Local files
# - Database
# - Cloud
# - IoT devices
# - Sensors
# - APIs
# ============================================================


# ============================================================
# 1. IMPORT REQUIRED LIBRARIES
# ============================================================

import os
import sys

# CustomException is used to handle errors in our project.
from src.exception import CustomException

# Logging is used to track what is happening in the pipeline.
from src.logger import logging

# Pandas is used to read and work with the dataset.
import pandas as pd

# train_test_split is used to divide the dataset into
# training data and testing data.
from sklearn.model_selection import train_test_split

# Dataclass is used to store configuration values
# in a clean and structured way.
from dataclasses import dataclass


# ============================================================
# 2. DATA INGESTION CONFIGURATION
# ============================================================

# This class stores all the file paths required
# during the data ingestion process.
#
# Instead of writing file paths again and again,
# we keep them in one place.

@dataclass
class DataIngestionConfig:

    # Path where the training dataset will be stored.
    train_data_path: str = os.path.join('artifacts', 'train.csv')

    # Path where the testing dataset will be stored.
    test_data_path: str = os.path.join('artifacts', 'test.csv')

    # Path where the original/raw dataset will be stored.
    raw_data_path: str = os.path.join('artifacts', 'raw.csv')


# ============================================================
# 3. DATA INGESTION CLASS
# ============================================================

# This class contains the complete data ingestion process.
#
# Main responsibility of this class:
#
# Data Source
#     ↓
# Read Data
#     ↓
# Save Raw Data
#     ↓
# Train-Test Split
#     ↓
# Save Train and Test Data

class DataIngestion:

    # --------------------------------------------------------
    # Constructor
    # --------------------------------------------------------
    # This method runs automatically when we create
    # an object of the DataIngestion class.

    def __init__(self):

        # Create the configuration object.
        #
        # This gives us access to:
        # - train_data_path
        # - test_data_path
        # - raw_data_path

        self.ingestion_config = DataIngestionConfig()


    # --------------------------------------------------------
    # DATA INGESTION METHOD
    # --------------------------------------------------------
    # This method performs the complete data ingestion process.

    def initiate_data_ingestion(self):

        # Log message to tell us that the data ingestion
        # process has started.

        logging.info("Entered the data ingestion method or component")

        try:

            # ------------------------------------------------
            # STEP 1: READ THE DATASET
            # ------------------------------------------------
            # Here we read the dataset from the local data folder.
            #
            # pd.read_csv() reads the CSV file and converts it
            # into a Pandas DataFrame.

            df = pd.read_csv(r'notebook\data\StudentsPerformance.csv')

            # Log message after successfully reading the dataset.

            logging.info("Read the dataset as Dataframe")


            # ------------------------------------------------
            # STEP 2: CREATE ARTIFACTS FOLDER
            # ------------------------------------------------
            # The artifacts folder will contain the output
            # files generated during the ML pipeline.
            #
            # If the folder already exists, exist_ok=True
            # prevents an error.

            os.makedirs(
                os.path.dirname(self.ingestion_config.train_data_path),
                exist_ok=True
            )


            # ------------------------------------------------
            # STEP 3: SAVE RAW DATA
            # ------------------------------------------------
            # Before modifying or splitting the dataset,
            # we save the original dataset as raw.csv.
            #
            # This helps us keep a copy of the original data.

            df.to_csv(
                self.ingestion_config.raw_data_path,
                index=False,
                header=True
            )


            # Log message before starting train-test split.

            logging.info("Train test split initiated")


            # ------------------------------------------------
            # STEP 4: SPLIT DATA INTO TRAIN AND TEST
            # ------------------------------------------------
            # We divide the complete dataset into two parts:
            #
            # 80% → Training data
            # 20% → Testing data
            #
            # random_state=42 makes the split reproducible.
            # It means we get the same split every time
            # we run the code.

            train_set, test_set = train_test_split(
                df,
                test_size=0.2,
                random_state=42
            )


            # ------------------------------------------------
            # STEP 5: SAVE TRAINING DATA
            # ------------------------------------------------
            # Save the training dataset inside the artifacts
            # folder as train.csv.

            train_set.to_csv(
                self.ingestion_config.train_data_path,
                index=False,
                header=True
            )


            # ------------------------------------------------
            # STEP 6: SAVE TESTING DATA
            # ------------------------------------------------
            # Save the testing dataset inside the artifacts
            # folder as test.csv.

            test_set.to_csv(
                self.ingestion_config.test_data_path,
                index=False,
                header=True
            )


            # Log message after the complete ingestion process
            # has been successfully completed.

            logging.info("Ingestion of the data is completed")


            # ------------------------------------------------
            # STEP 7: RETURN OUTPUT FILE PATHS
            # ------------------------------------------------
            # Return the locations of train.csv and test.csv.
            #
            # These paths can be used by the next component
            # of the ML pipeline.

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )


        # ----------------------------------------------------
        # ERROR HANDLING
        # ----------------------------------------------------
        # If any error happens inside the try block,
        # the error will be caught here.

        except Exception as e:

            # Convert the normal error into our project's
            # custom exception and pass the system information.

            raise CustomException(e, sys)


# ============================================================
# 4. MAIN PROGRAM
# ============================================================
# This block runs only when this Python file is executed
# directly.
#
# It will not run automatically if this file is imported
# into another Python file.

if __name__ == "__main__":

    # Create an object of the DataIngestion class.

    obj = DataIngestion()

    # Start the complete data ingestion process.

    obj.initiate_data_ingestion()