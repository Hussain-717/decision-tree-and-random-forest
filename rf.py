import streamlit as st
import json 
import pandas as pd
df = pd.read_csv("ODI_Match_info.csv")
with open('metrics.json', 'r') as f:
    metrics = json.load(f)

st.title("Comparsion between Decision tree and Random Forest")
st.write("Few columns of the the data set looks like: ")
st.write(df.head())
st.write("Accuracy of Decision tree: ")
st.write(f"Training Accuracy: {metrics['Training Accuracy dt']:.4f}")
st.write(f"Testing Accuracy: {metrics['Testing Accuracy dt']:.4f}")
st.write("Accuracy of Random Forest: ")
st.write(f"Training Accuracy: {metrics['Training Accuracy rf']:.4f}")
st.write(f"Testing Accuracy: {metrics['Testing Accuracy rf']:.4f}")
