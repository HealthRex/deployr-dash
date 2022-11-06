import streamlit as st
from PIL import Image

# Custom imports 
from utils.multipage import MultiPage
from pages import (deployr)

st.set_page_config(layout="wide")

app = MultiPage()

# Add all your applications (pages) here
app.add_page("Deployr", deployr.app)

app.run()