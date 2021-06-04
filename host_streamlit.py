# Imports
import pandas as pd
import streamlit as st
from semantic_role_labeling import srl_model


# Main Streamlit UI
"""
# Natural Language Understanding
"""

sentence = st.text_area("Enter Text:", "Rajasekar built this tool to understand the segments of the text. Using this, we can gain some insights from the data.")
df, result = srl_model.get_predictions(sentence.strip())
st.write(df)

