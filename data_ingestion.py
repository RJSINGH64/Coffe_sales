import pandas as pd
import numpy as np 
from config import mongo_client #mongo url 
from dataclasses import dataclass 
from logger import logging
from exception import ProjectException
import os , sys

@dataclass
class DataIngestion:
    
    def initiate_data_ingestion(self):

        try:  
           logging.info(f"Loading Data from MongoDB Database")
           df = pd.DataFrame(mongo_client['Coffee']["Sales"].find())
           logging.info(f"Rows {df.shape[0]} Columns {df.shape[1]} Available after droping _id" )
           if "_id" in df.columns:
               df.drop("_id" , axis=1 , inplace=True)
           artifact_feature_store="artifacts/feature_store"
           os.makedirs(artifact_feature_store, exist_ok=True)
           logging.info(f'Initiate data ingestion Pipeline')
           artifact_file_path=os.path.join(artifact_feature_store , "coffe_sales.csv")
           df.to_csv(artifact_file_path , index=False)
           logging.info(f"Saving Coffee-Sales CSV file to feature_store folder")
        
        except Exception as e:
           raise ProjectException(e , sys) 
    

   

    
               

          


