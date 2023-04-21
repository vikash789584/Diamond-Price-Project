import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            logging.info('Data Transformation initiated')
            # Define which columns should be ordinal-encoded and which should be scaled
            categorical_cols = ['Order_Date', 'Time_Orderd', 'Time_Order_picked', 'Weather_conditions',
                                'Road_traffic_density', 'Type_of_order', 'Type_of_vehicle', 'Festival',
                                'City']
            numerical_cols = ['Delivery_person_Age', 'Delivery_person_Ratings', 'Restaurant_latitude',
                              'Restaurant_longitude', 'Delivery_location_latitude',
                              'Delivery_location_longitude', 'Vehicle_condition','multiple_deliveries']
            
            # Define the custom ranking for each ordinal variable
            Order_Date_categories = ["11-02-2022","12-02-2022","13-02-2022","14-02-2022",
                                     "15-02-2022","16-02-2022","17-02-2022","18-02-2022",
                                     "01-03-2022","02-03-2022","03-03-2022","04-03-2022","05-03-2022",
                                     "06-03-2022","07-03-2022","08-03-2022","09-03-2022","10-03-2022",
                                     "11-03-2022","12-03-2022","13-03-2022","14-03-2022","15-03-2022",
                                     "16-03-2022","17-03-2022","18-03-2022","19-03-2022","20-03-2022",
                                     "21-03-2022","23-03-2022","24-03-2022","25-03-2022","26-03-2022",
                                     "27-03-2022","28-03-2022","29-03-2022","30-03-2022","31-03-2022",
                                     "01-04-2022","02-04-2022","03-04-2022","04-04-2022","05-04-2022","06-04-2022"]
            Time_Orderd_categories = ["21:55","14:55","17:30","09:20", "19:50","20:25","20:30",
                                      "20:40", "21:15","20:20","22:30","08:15","19:30","12:25",
                                      "18:35", "20:35", "23:20", "21:20", "23:35", "22:35", "23:25",
                                      "13:35", "21:35", "18:55", "14:15", "0.458333333", "09:45",
                                      "08:40", "0.958333333", "17:25", "nan", "19:45", "19:10", "10:55",
                                      "21:40", "0.791666667", "16:45", "11:30", "15:10", "22:45",
                                      "22:10", "20:45", "22:50", "17:55", "09:25", "20:15", "22:25",
                                      "22:40", "23:50", "15:25", "10:20", "20:55", "10:40", "15:55",
                                      "20:10", "12:10", "15:30", "10:35", "21:10", "20:50", "12:35",
                                      "0.875", "23:40", "18:15", "18:20", "11:45", "12:45", "23:30",
                                      "10:50", "21:25", "10:10", "17:50", "22:20", "12:40", "23:55",
                                      "10:25", "08:45", "23:45", "19:55", "22:15", "23:10", "09:15",
                                      "18:25", "18:45", "16:50", "1", "14:20", "10:15", "08:50", "0.375",
                                      "17:45", "16:35", "08:30", "21:45", "19:40", "14:50", "18:10",
                                      "12:20", "12:50", "09:10", "12:30", "17:10", "19:15", "17:20",
                                      "18:30", "13:10", "19:35", "09:50", "0.625", "0.833333333",
                                      "10:30", "09:40", "15:35", "16:55", "22:55", "0.666666667", "0.75",
                                      "17:15", "21:30", "18:40", "11:10", "13:50", "0.416666667",
                                      "21:50", "11:50", "13:30", "0.916666667", "08:25", "11:20",
                                      "11:55", "09:30", "08:20", "08:10", "11:40", "23:15", "19:20",
                                      "12:15", "11:35", "11:15", "17:35", "17:40", "14:40", "18:50",
                                      "11:25", "14:25", "0.5", "16:10", "19:25", "08:55", "13:40",
                                      "0.708333333", "09:35", "08:35", "16:15", "13:20", "15:50",
                                      "15:20", "16:20", "14:30", "15:45", "16:40", "0.541666667",
                                      "12:55", "10:45", "13:25", "09:55", "15:15", "13:15",
                                      "0.583333333", "15:40", "16:25", "14:10", "13:45", "13:55",
                                      "14:35", "16:30", "14:45"]
            Time_Order_picked_categories = ["22:10", "15:05", "17:40", "09:30", "20:05", "20:35", "15:10",
                                            "20:40", "20:50", "21:30", "20:25", "22:45", "08:30", "19:45",
                                            "12:30", "18:50", "23:30", "21:35", "23:45", "22:50", "22:40",
                                            "23:35", "13:40", "21:45", "19:10", "14:25", "11:10", "09:55",
                                            "08:55", "23:10", "17:30", "18:35", "19:50", "19:25","0.458333333",
                                            "19:15", "16:55", "11:40", "15:15", "22:55","22:25", "20:55","23:05",
                                            "0.75", "0.958333333", "09:40", "20:20","22:35","0.916666667", "23:55",
                                            "15:40", "10:30", "0.875","10:50", "16:05","20:15", "12:15", "15:45",
                                            "22:15", "10:45","15:30", "24:05:00", "21:25", "12:45", "21:15", "18:20",
                                            "18:25","11:50", "12:50", "10:55", "21:40", "10:20", "17:55", "23:50",
                                            "12:55", "24:10:00", "10:40", "0.375", "20:45", "0.833333333",
                                            "23:15", "22:20", "21:05", "0.708333333", "24:15:00", "21:20",
                                            "14:35", "10:25", "09:05", "16:50", "08:40", "23:40", "21:50",
                                            "19:55", "0.625", "10:35", "09:25", "17:20", "19:30", "17:25",
                                            "20:10", "1", "17:35", "0.791666667", "19:05", "13:20", "17:50",
                                            "18:05", "19:20", "10:05", "09:10", "21:55", "19:40", "18:10",
                                            "09:50", "15:50", "18:30", "18:15", "16:15", "11:15", "21:10",
                                            "22:30", "15:20", "18:40", "23:20", "11:25", "13:55", "18:45",
                                            "22:05", "11:55", "18:55", "09:45", "17:15", "12:05", "0.5",
                                            "19:35", "08:25", "11:05", "15:35", "12:40", "12:25", "08:20",
                                            "23:25", "16:10", "13:45", "08:15", "08:45", "20:30","0.541666667",
                                            "11:20", "08:50", "14:45", "17:45", "0.416666667","08:35", "12:10",
                                            "11:35", "14:30", "10:15", "17:05", "10:10","09:35", "11:30", "16:25",
                                            "09:15", "13:35", "15:55", "13:10","13:05", "11:45", "16:20", "16:30", 
                                            "16:45", "09:20", "13:25","14:15", "14:05", "16:35", "16:40", "13:30",
                                            "14:40", "12:20","13:50", "17:10", "15:25", "14:20", "12:35", "0.583333333",
                                            "14:10", "14:55", "13:15", "14:50", "0.666666667"]
            Weather_conditions_categories=["Fog","Stormy","Sandstorms","Windy","Cloudy","Sunny"]
            Road_traffic_density_categories=["Low","Medium","High","Jam"]
            Type_of_order_categories=["Snack","Meal","Drinks","Buffet"]
            Type_of_vehicle_categories=["motorcycle","scooter","electric_scooter","bicycle"]
            Festival_categories=["Yes","No"]
            City_categories=["Metropolitian", "Urban", "Semi-Urban"]
            
            logging.info('Pipeline Initiated')

            ## Numerical Pipeline
            num_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='median')),
                ('scaler',StandardScaler())

                ]

            )

            # Categorigal Pipeline
            cat_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('ordinalencoder',OrdinalEncoder(categories=[Order_Date_categories,Time_Orderd_categories,
                                                             Time_Order_picked_categories,Weather_conditions_categories,
                                                             Road_traffic_density_categories,Type_of_order_categories,
                                                             Type_of_vehicle_categories,Festival_categories,
                                                             City_categories])),
                ('scaler',StandardScaler())
                ]

            )

            preprocessor=ColumnTransformer([
            ('num_pipeline',num_pipeline,numerical_cols),
            ('cat_pipeline',cat_pipeline,categorical_cols)
            ])
            
            return preprocessor

            logging.info('Pipeline Completed')

        except Exception as e:
            logging.info("Error in Data Trnasformation")
            raise CustomException(e,sys)
        
    def initaite_data_transformation(self,train_path,test_path):
        try:
            # Reading train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data completed')
            logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head  : \n{test_df.head().to_string()}')

            logging.info('Obtaining preprocessing object')

            preprocessing_obj = self.get_data_transformation_object()

            target_column_name = 'Time_taken (min)'
            drop_columns = [target_column_name,'ID','Delivery_person_ID']

            input_feature_train_df = train_df.drop(columns=drop_columns,axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=drop_columns,axis=1)
            target_feature_test_df=test_df[target_column_name]
            
            ## Trnasformating using preprocessor obj
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            logging.info("Applying preprocessing object on training and testing datasets.")
            

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )
            logging.info('Preprocessor pickle file saved')

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
            
        except Exception as e:
            logging.info("Exception occured in the initiate_datatransformation")

            raise CustomException(e,sys)