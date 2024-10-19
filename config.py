from dataclasses import dataclass
from logger import logging 
from utils.exception import ProjectException
from dotenv import load_dotenv
import pymongo as pm
import os , sys

#loading .env file
print(f"{'>>'*10} loading  .env {'<<'*10}")
logging.info(f"{'>>'*10} loading  .env {'<<'*10}")
load_dotenv()


@dataclass
class Environmentvariable:
    mongo_url:str = os.getenv("MONGO_DB_URL")

#creating Fuction for establish connection
def mongo_connect(): 
    try:
        obj = Environmentvariable()  #instance for url
        MONGO_CLIENT= pm.MongoClient(obj.mongo_url)
        print(f"{'>>'*10} Sucessfully Connected to  MongoDB Database {'<<'*10}")
        logging.info(f"{'>>'*10} Sucessfully Connected to  MongoDB Database {'<<'*10}")
        return MONGO_CLIENT 
     
    except Exception as e:
        raise ProjectException(e , sys) 
        logging.info(e , sys)
    
mongo_client= mongo_connect()   #creating instance for MongoDb connection 
