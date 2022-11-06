import streamlit as st
from datetime import datetime, timedelta

def render_date_selector():
    last_day_of_prev_month = datetime.today().replace(day=1) - timedelta(days=1)
    start_day_of_prev_month = datetime.today().replace(day=1) - timedelta(days=last_day_of_prev_month.day)
    start_day_of_current_month = datetime.today().replace(day=1)
    
    start_date = st.sidebar.date_input('Start date', start_day_of_prev_month, min_value=datetime(2021, 10, 1), max_value=datetime.today().date())
    end_date = st.sidebar.date_input('End date', start_day_of_current_month, start_date, max_value=datetime.today().date())
    
    if start_date < end_date:
        st.sidebar.success('Start date: `%s`\n\nEnd date:`%s`' % (start_date, end_date))
    else:
        st.sidebar.error('Error: End date must fall after start date.')

    return start_date, end_date