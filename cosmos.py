import azure.cosmos.cosmos_client as cosmos_client
import os
import pandas as pd

import pdb

def read_predictions_and_labels(
    from_date='2022-10-28',
    to_date='2022-10-30',
    container='test_predictions_container',
    model='20220705_label_HCT_deploy.pkl'):
    """
    Grab a batch of json inferences and reads in as dataframe
    
    Args:
        container : str specifying container where json items are stored
        model : str specifying which model inference to read
        from_date : grab jsons with inference time >= this date
        to_date : grab jsosn with inference time <= this date
    Returns:
        df : dataframe rep of json items containing model predictions and labels
    """
    client = cosmos_client.CosmosClient(
        os.environ['COSMOS_HOST'],
        os.environ['COSMOS_READ_KEY']
    )
    db = client.get_database_client(os.environ['COSMOS_DB_ID'])
    container = db.get_container_client(container)
    query = f"""
    SELECT 
        *
    FROM 
        c 
    WHERE 
        c.model = '{model}'
    AND
        c.inference_time >= '{from_date}' AND
        c.inference_time <= '{to_date}'
    """

    items = list(container.query_items(
        query=query,
    ))
    df = pd.DataFrame.from_records(items)
    return df
