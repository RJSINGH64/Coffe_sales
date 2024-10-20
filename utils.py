from config import mongo_client 
from logger import logging
from exception import ProjectException
import pandas as pd
import numpy as np
import os , sys
from scipy.stats import ks_2samp
from typing import Optional

def get_dataframe_from_mongoDb(database,collection):
     
     """ 
     ======================================================================
      
      this funtion return data as pd.Dataframe , extract data from MongoDb.
      funtion also remove unnecessary column specially _id from dataframe.
      
      database : str = Database 
      collection : str = Collection name
      
      ======================================================================
     
     """
     try:
          logging.info(f"from Database {database} Collection {collection} importing data")
          dataframe=pd.DataFrame(list(mongo_client[database][collection].find()))
          logging.info(f"Rows : {dataframe.shape[0]} column : {dataframe.shape[1]}")
          logging.info(f"These Columns are Available  {dataframe.columns}" ) 
          if "_id" in dataframe.columns:
               dataframe.drop("_id" , axis=1 , inplace=True)
          logging.info(f"Unecessary column _id removed")
          return dataframe  
     
     except Exception as e:
          raise ProjectException(e ,sys)
     

