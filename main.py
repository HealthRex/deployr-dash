import streamlit as st
from PIL import Image

# Custom imports 
from utils.multipage import MultiPage
from pages import (
    hematocrit, 
    hemoglobin,
    platelets,
    white_blood_cells,
    albumin,
    blood_urea_nitrogen,
    calcium,
    carbon_dioxide,
    creatinine,
    potassium,
    sodium,
    magnesium
    )

st.set_page_config(layout="wide")

app = MultiPage()

# Add all your applications (pages) here
app.add_page("Hematocrit", hematocrit.app)
app.add_page("Hemoglobin", hemoglobin.app)
app.add_page("Platelets", platelets.app)
app.add_page("White_blood_cells", white_blood_cells.app)
app.add_page("Albumin", albumin.app)
app.add_page("Blood_urea_nitrogen", blood_urea_nitrogen.app)
app.add_page("Calcium", calcium.app)
app.add_page("Carbon_dioxide", carbon_dioxide.app)
app.add_page("Creatinine", creatinine.app)
app.add_page("Potassium", potassium.app)
app.add_page("Sodium", sodium.app)
app.add_page("Magnesium", magnesium.app)

app.run()