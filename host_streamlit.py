# Imports
import pandas as pd
import streamlit as st
from semantic_role_labeling import srl_model


# Main Streamlit UI
"""
# Natural Language Understanding
"""

args2meaning_map = {'ARG0': 'Causer', 
                    'ARG1': 'Affected', 
                    'ARG2': 'Effect', 
                    'ARG3': 'Start Point', 
                    'ARG4': 'End Point',
                    'ARGM-TMP': 'When', 
                    'ARGM-LOC': 'Where', 
                    'ARGM-DIR': 'From/To', 
                    'ARGM-MNR': 'How', 
                    'ARGM-PRP': 'Why1', 
                    'ARGM-CAU': 'Why2', 
                    'ARGM-REC': 'Whom', 
                    'ARGM-ADV': 'Miscellaneous', 
                    'ARGM-PRD': 'Secondary Predicate', 
                    'V': 'Verb', 
                    'DESCRIPTION': 'Description', 
                    'SENTENCE': 'Sentence', 
                    'INDEX': 'Sentence Index'}

sentence = st.text_area("Enter Text:", "Rajasekar built this tool to understand the segments of the text. Using this, we can gain some insights from the data.")
df, result = srl_model.get_predictions(sentence.strip())
# df.columns = args2meaning_map.values()
df.columns = [args2meaning_map.get(item) for item in df.columns.tolist()]
st.write(df)

