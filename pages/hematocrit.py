import streamlit as st
import altair as alt
import json
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from PIL import Image
import seaborn as sns

from os import getenv
from datetime import datetime

from utils.cosmos import read_predictions_and_labels
from utils.evaluators import BinaryEvaluator, BinaryGroupEvaluator
from utils.constants import *

from components.csv_downloader import render_csv_downloader
from components.date_selector import render_date_selector

import warnings
warnings.filterwarnings('ignore')
print("No Warning Shown")

import pdb

def mock_get_hct_data(start_date, end_date):
    pass

def get_hct_feature_order():
    with open(os.path.join(HCT_OUTPUT_DIR, '20230105_label_HCT_deploy.pkl'), 'rb') as f:
        model = pickle.load(f)
    df_feature_order = pd.DataFrame(data={
        'indices' : [i for i in range(len(model['feature_order']))],
        'features' : model['feature_order']
    })
    return df_feature_order

def get_race_group(non_zero_features, race_map):
    for ind in race_map:
        if str(ind) in non_zero_features:
            return race_map[ind]
    print("race missing")
    return 'race_missing'

def add_feature_occurances(df, df_feature_order, features=[]):
    feature_map = {f : ind for f, ind in 
                   zip(df_feature_order.features, df_feature_order.indices)}
    for feature in features:
        try:
            df[feature] = [float(f[str(feature_map[feature])])
                        if str(feature_map[feature]) in f 
                        else 0 for f in df.feature_vector]
        except:
            pdb.set_trace()
    return df
    
def build_group_dataframe(df, df_feature_order):
    """
    Appends group information to df 
    """
    df_race = df_feature_order[df_feature_order['features'].str.contains('race_')].reset_index()
    race_map = {ind : race for ind, race in zip(df_race.indices, df_race.features)}
    df['group'] = [get_race_group(f, race_map) for f in df.feature_vector]
    return df

def get_hct_data(start_date, end_date):
    df = read_predictions_and_labels(start_date, 
                                     end_date,
                                     model='20230105_label_HCT_deploy.pkl')

    # Drop missing - complete case only
    df = df[df['label'] != -1]

    # Get feature order
    df_feature_order = get_hct_feature_order()

    # Get group data
    df = build_group_dataframe(df, df_feature_order)

    evalr = BinaryEvaluator(outdir=HCT_OUTPUT_DIR)
    evalr(df.label, df.prediction)

    # Group evalr
    # group_evalr = BinaryGroupEvaluator(outdir=HCT_OUTPUT_DIR)
    # group_evalr(df)

    # Get feature data
    df = add_feature_occurances(df, df_feature_order, features=['Age_1', 'Age_2', 'sex_Male'])

    # Create feature and label distribution figures
    fig, axs = plt.subplots(2, 2, figsize=(40, 20))
    value_names = ['label', 'Age_1', 'Age_2', 'sex_Male']
    display_names = {'label' : 'Class label',
                     'Age_1' : 'Age in 20-40th percentile',
                     'Age_2' : 'Age in 40-60th percentile',
                     'sex_Male' : 'Sex is male'}
    row, col = 0, 0
    dates = [d[:10] for d in df['inference_time'].values]
    for i in range(4):
        plot_mean_value_by_time(df[value_names[i]].values,
                                dates,
                                value_name=display_names[value_names[i]],
                                ax=axs[row, col])
        if col < 1:
            col += 1
        else:
            row += 1
            col = 0
    fig.subplots_adjust(hspace=0.4)
    plt.savefig(os.path.join(HCT_OUTPUT_DIR, 'feature_and_label_dist.png'),
                bbox_inches='tight', dpi=300)


def plot_mean_value_by_time(values, dates, value_name, ax):
    df_vals = pd.DataFrame(data={
         value_name : values,
        'date' : dates 
    })
    df_vals = df_vals.sort_values('date')
    sns.barplot(
        data=df_vals,
        x='date',
        y=value_name,
        ax=ax,
        color='crimson'
    )
    ax.set_title(f"{value_name} average across examples")
    ax.set_ylabel('')
    ax.set_xlabel('')
    ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=90, ha='right')


def render_hct_results(col):
    sns.set_theme(style='whitegrid', font_scale=2.0)
    with open(os.path.join(HCT_OUTPUT_DIR, 'metrics.json'), 'r') as f:
        metrics = json.load(f)

    metrics_df = pd.DataFrame({"Metrics": metrics.keys(), "Values": metrics.values()})
    image = Image.open(os.path.join(HCT_OUTPUT_DIR, "performance_curves.png"))

    # st.header("Global model peformance")
    my_expander = st.expander("Global model peformance", expanded=True)
    with my_expander:
        st.table(metrics_df)
        st.image(image, caption='Hematocrit Performance Curves')

    # st.header("Group fairness evaluation")
    my_expander2 = st.expander("Group fairness evaluation", expanded=True)
    with my_expander2:
        group_image = Image.open(os.path.join(HCT_OUTPUT_DIR, "group_performance_curves.png"))
        st.image(group_image, caption="Hematocrit group performance curves")
    

    my_expander3 = st.expander("Feature and label distributions", expanded=True)
    with my_expander3:
        # clicked = my_widget("second")

    # st.header("Feature and label distributions")
        feature_image = Image.open(os.path.join(HCT_OUTPUT_DIR, "feature_and_label_dist.png"))
        st.image(feature_image, caption="Hematocrit feature and label distributions")

def app():

    start_date, end_date = render_date_selector()
    start_date = str(start_date)[:10]
    end_date = str(end_date)[:10]
    # start_date = '2023-01-11'
    # end_date = '2023-01-12'

    page_columns = st.columns((7, 7))

    get_hct_data(start_date, end_date)
    render_hct_results(page_columns[0])









