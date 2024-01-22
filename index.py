import streamlit as st
from streamlit.components import v1 as components

st.set_page_config(layout="wide")

# Load HTML file
with open('index.html', 'r') as file:    
    html_content = file.read()

components.html(html_content, width = None, height=2300)