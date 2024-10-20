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
           return df   
        except Exception as e:
           raise ProjectException(e , sys) 

    
               

          


