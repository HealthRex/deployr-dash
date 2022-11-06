import streamlit as st
from PIL import Image

class MultiPage:
    """Combines many streamlit applications into one class"""

    def __init__(self) -> None:
        "List to store all appliations as an instance variable"
        self.pages = []

    def add_page(self, title, function) -> None:

        self.pages.append({
            "title": title, 
            "function": function
        })

    def run(self):

        image = Image.open('assets/stanford_med_logo.jpg')
        st.sidebar.image(image, width=300)

        page = st.sidebar.selectbox(
            '', 
            self.pages, 
            format_func=lambda page: page['title']
        )

        page['function']()


