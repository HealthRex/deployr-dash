import streamlit as st

@st.cache
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

def render_csv_downloader(data, file_name, help_info=''):
    if data is not None:
        csv = convert_df(data)
        
        st.sidebar.download_button(label="Download data as CSV",
                            data=csv,
                            file_name=file_name,
                            mime='text/csv',
                            help=help_info)