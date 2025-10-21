import os
import sys
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging

# Resolve project root from this file's location: src/components/ -> src -> project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

@dataclass
class DataIngestionConfig:
    artifacts_dir: str = os.path.join(PROJECT_ROOT, 'artifacts')
    train_data_path: str = os.path.join(PROJECT_ROOT, 'artifacts', 'train.csv')
    test_data_path: str  = os.path.join(PROJECT_ROOT, 'artifacts', 'test.csv')
    raw_data_path: str   = os.path.join(PROJECT_ROOT, 'artifacts', 'data.csv')
    source_csv_path: str = os.path.join(PROJECT_ROOT, 'notebook', 'data', 'stud.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info('Entered the data ingestion method or component')
        try:
            # Ensure artifacts directory exists
            os.makedirs(self.ingestion_config.artifacts_dir, exist_ok=True)

            # Read source data from absolute, OS-agnostic path
            src_path = self.ingestion_config.source_csv_path
            logging.info(f'Reading data from {src_path}')
            df = pd.read_csv(src_path)
            logging.info('Read the data from file')

            # Save raw copy
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            # Split
            logging.info('Train Test split initialized')
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Save splits
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info('Train and Test split completed')

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
                self.ingestion_config.raw_data_path,
            )
        except Exception as e:
            # Log for context, then raise
            logging.exception('Data ingestion failed')
            raise CustomException(e, sys)

if __name__ == '__main__':
    obj = DataIngestion()
    obj.initiate_data_ingestion()
