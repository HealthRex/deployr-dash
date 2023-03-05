import streamlit as st
import altair as alt
import json
import pandas as pd
from PIL import Image

from os import getenv
from datetime import datetime

from utils.cosmos import read_predictions_and_labels
from utils.evaluators import BinaryEvaluator
from utils.constants import *

from components.csv_downloader import render_csv_downloader
from components.date_selector import render_date_selector

import warnings
warnings.filterwarnings('ignore')
print("No Warning Shown")

def mock_get_hct_data(start_date, end_date):
    pass

def get_hct_data(start_date, end_date):
    df = read_predictions_and_labels(start_date, 
                                     end_date,
                                     model='20230105_label_PLT_deploy.pkl')

    # Drop missing - complete case only
    df = df[df['label'] != -1]

    evalr = BinaryEvaluator(outdir=HCT_OUTPUT_DIR)
    evalr(df.label, df.prediction)


def render_hct_results(col):
    with open(os.path.join(HCT_OUTPUT_DIR, 'metrics.json'), 'r') as f:
        metrics = json.load(f)

    metrics_df = pd.DataFrame({"Metrics": metrics.keys(), "Values": metrics.values()})
    image = Image.open(os.path.join(HCT_OUTPUT_DIR, "performance_curves.png"))

    st.header("Global model peformance")
    st.table(metrics_df)
    st.image(image, caption='Platelets Performance Curves')

def app():

    start_date, end_date = render_date_selector()
    start_date = str(start_date)[:10]
    end_date = str(end_date)[:10]

    page_columns = st.columns((7, 7))

    get_hct_data(start_date, end_date)
    render_hct_results(page_columns[0])









