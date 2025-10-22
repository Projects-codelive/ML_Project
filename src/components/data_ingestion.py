import os
import sys
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformation

# Resolve project root: src/components -> src -> project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

@dataclass
class DataIngestionConfig:
    artifacts_dir: str = os.path.join(PROJECT_ROOT, 'artifacts')
    train_data_path: str = os.path.join(PROJECT_ROOT, 'artifacts', "train.csv")
    test_data_path: str = os.path.join(PROJECT_ROOT, 'artifacts', "test.csv")
    raw_data_path: str = os.path.join(PROJECT_ROOT, 'artifacts', "data.csv")
    source_csv_path: str = os.path.join(PROJECT_ROOT, 'notebook', 'data', 'stud.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            # Ensure artifacts directory exists
            os.makedirs(self.ingestion_config.artifacts_dir, exist_ok=True)

            # Read source CSV from absolute, normalized path
            src_path = self.ingestion_config.source_csv_path
            if not os.path.exists(src_path):
                raise FileNotFoundError(f"Source CSV not found at {src_path}")
            df = pd.read_csv(src_path)
            logging.info('Read the dataset as dataframe')

            # Save raw copy
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            # Split and save
            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the data is completed")

            # Return exactly a 2-tuple
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()
    data_transformation = DataTransformation()
    train_arr, test_arr, preproc_path = data_transformation.initiate_data_transformation(train_data, test_data)
    logging.info(f"Preprocessor saved to: {preproc_path}")
