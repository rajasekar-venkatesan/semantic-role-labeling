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

sentence = st.text_area("Enter Text:", "My SQ experience is bad because the flight was delayed by 2 hours.")
df, result = srl_model.get_predictions(sentence.strip())
# df.columns = args2meaning_map.values()
df.columns = [args2meaning_map.get(item) for item in df.columns.tolist()]
df = df.iloc[:, :-2]
# st.write(df)
result = df.to_dict(orient='records')
result = [{k: v for k, v in item.items() if v} for item in result]
st.write(result)

