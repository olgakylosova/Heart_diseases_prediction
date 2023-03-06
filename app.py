import pickle
import streamlit as st


st.header('Heart diseases prediction')

def load ():
    with open("/Users/olgakylosova/PycharmProjects/Heart diseases prediction/model.pcl", "rb") as fid:
        return pickle.load(fid)

lc, rc = st.columns(2)
with lc:
    gender = st.radio("Gender (1 = female, 2 = male)", (1, 2))
with rc:
    age = st.number_input('Age', 0, 100, key='age')

height = st.slider('Height', 140, 200, key='height')
weight = st.slider('Weight', 40, 120, key='weight')
ap_hi = st.slider('Ap_hi', 100, 170, key='ap_hi')
ap_lo = st.slider('Ap_lo', 60, 105, key='ap_lo')

lc, rc = st.columns(2)
with lc:
    cholesterol = st.selectbox("Cholesterol", [1, 2, 3], key="cholesterol")
with rc:
    gluc = st.selectbox("Gluc", [1, 2, 3], key="gluc")

lc, mc, rc = st.columns(3)
with lc:
    smoke = st.radio("Smoke (0 = no, 1 = yes)", (0, 1))
with mc:
    alco = st.radio("Alco (0 = no, 1 = yes)", (0, 1))
with rc:
    active = st.radio("Active (0 = no, 1 = yes)", (0, 1))

st.caption('_Probability heart diseases_')
model = load()

probabilities_one_test = model.predict
model.predict_proba([[age, height, weight, ap_hi, ap_lo, gender, cholesterol, gluc, smoke, alco, active]])[:,1]

st.write(probabilities_one_test)
