from src.components.Data_Ingestion import DataIngestion
from src.components.Data_Tranformation import DataTransformation
from src.components.Model_Trainer import ModelTrainer






if __name__ == "__main__":
    Data_Ingestion = DataIngestion()
    train_data_path,test_data_path = Data_Ingestion.initiate_data_ingestion()

    Data_transformation= DataTransformation()
    train_array,test_array,_= Data_transformation.initiate_data_transformation(train_data_path,test_data_path)

    Model_Trainer= ModelTrainer()
    print(Model_Trainer.initiate_model_trainer(train_array = train_array,
                                         test_array= test_array))

