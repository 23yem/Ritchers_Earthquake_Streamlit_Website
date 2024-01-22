import tensorflow as tf
import streamlit as st
from streamlit.components import v1 as components
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
import joblib 


from data_preprocessing_XGBoost import data_preprocessing

st.set_page_config(layout="wide")

# Load part 1
with open('LGBM_1.html', 'r') as file:    
    html_content_1 = file.read()

components.html(html_content_1, width = None, height=1000)




# Load part 2
with open('LGBM_2.html', 'r') as file:    
    html_content_2 = file.read()

components.html(html_content_2, width = None, height=6000)