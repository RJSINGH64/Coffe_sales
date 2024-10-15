from config import mongo_client # importing logging from  my Project Package
import pandas as pd
from utils.exception import ProjectException  # importing  Execption from  my Project Package
from logger import logging
import os ,sys

database= "Coffee" #var for database name
collection_name = "Sales"#var collection name

#file path using os 
file_path = os.path.join(os.getcwd() , "dataset\Cofee Sales dataset.csv") 
dataframe = pd.read_csv(file_path) #dataframe 

#Dumping data into a MongoDB Atlas
if __name__=="__main__":

    try:
        #Converting dataframe into dictionary , because Mongo only store data as key and Values format such as json and dict.
        dict_data=dataframe.to_dict(orient="records") #data as dictionary records
        mongo_client[database][collection_name].insert_many(dict_data) #dumping dictionary data into MongoDb
        logging.info(f">>> Dataset Sucessfully Stored inside MongoDb Database <<<")
        print(f">>> Dataset Sucessfully Stored inside MongoDb Database <<<")
    except Exception as e:
        raise ProjectException(e , sys)  #Exception while  any error
        logging.info(e , sys)