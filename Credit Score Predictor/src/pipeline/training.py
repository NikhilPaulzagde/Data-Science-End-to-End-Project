from src.components.Data_Ingestion import DataIngestion
from src.components.Data_Transformation import DataTransformation
from src.components.Model_Trainer import ModelTrainer

if __name__ == "__main__":
    DataIngestion = DataIngestion()
    train_data, test_data =DataIngestion.initiate_data_ingestion()

    DataTransformation = DataTransformation()
    train_arr,test_arr,_ = DataTransformation.initiate_data_transformation(train_data, test_data)

    ModelTrainer = ModelTrainer()
    print(ModelTrainer.initiate_model_trainer(train_arr,test_arr))