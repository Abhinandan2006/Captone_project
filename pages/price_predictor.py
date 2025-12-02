import pandas as pd
import streamlit as st
import pickle

df = pd.read_csv('data/final_df.csv')
st.dataframe(df)