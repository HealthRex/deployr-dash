import azure.cosmos.cosmos_client as cosmos_client
import os
import pandas as pd
from utils.constants import *

import pdb

def read_predictions_and_labels(
    start_date='2023-01-11',
    end_date='2023-01-14',
    container='20230218_labels_and_predictions',
    model='20230105_label_HCT_deploy.pkl'):
    """
    Grab a batch of json inferences and reads in as dataframe
    
    Args:
        container : str specifying container where json items are stored
        model : str specifying which model inference to read
        start_date : grab jsons with inference time >= this date
        end_date : grab jsosn with inference time <= this date
    Returns:
        df : dataframe rep of json items containing model predictions and labels
    """

    client = cosmos_client.CosmosClient(
        COSMOS_HOST, 
        COSMOS_READ_KEY
    )
    db = client.get_database_client(COSMOS_DB_ID)
    container = db.get_container_client(container)
    query = f"""
    SELECT 
        *
    FROM 
        c 
    WHERE 
        c.model = '{model}'
    AND
        c.inference_time >= '{start_date}' AND
        c.inference_time <= '{end_date}'
    """

    items = list(container.query_items(
        query=query,
    ))
    df = pd.DataFrame.from_records(items)
    df = df[df['label'] != -1].reset_index() # complete case
    df['time_to_result'] = (pd.to_datetime(df['result_time'])
         - pd.to_datetime(df['inference_time']))
    df = df[df['time_to_result'].dt.days < 1].reset_index() # result in 2 days
    return df

if __name__ == '__main__':
    df = read_predictions_and_labels()
    pdb.set_trace()
