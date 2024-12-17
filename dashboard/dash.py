# Import Library
import itertools
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from my_model import MyModel


# LOAD DATA
# ==============================
def load_data():
    data = pd.read_csv("data/breast-cancer-wisconsin.data")
    return data

data = load_data()

# Mengganti nama kolom
data.columns = ['ID', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape', 
                'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 
                'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class']

# Menghapus kolom ID dan menangani nilai yang hilang
data = data.drop(columns=['ID'])
data.replace('?', np.NaN, inplace=True)
data.dropna(inplace=True)

# ========================================================================================================================================================================================

# STREAMLIT
st.set_page_config(
  page_title = "Breast Cancer Detection",
  page_icon = ":heart:"
)

# TITLE DASHBOARD
# ==============================
st.title("Breast Cancer Detection")
st.title("Accuracy : 96%")
st.write("")


# SIDEBAR
# ==============================
st.sidebar.title("Informasi Data:")
st.sidebar.markdown("This breast cancer databases was obtained from the University of Wisconsin Hospitals, Madison from Dr. William H. Wolberg.  ")
# Show dataset source
st.sidebar.markdown("[Download Dataset](https://archive.ics.uci.edu/dataset/15/breast+cancer+wisconsin+original)")

# Form input
features = ['Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape', 
            'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 
            'Bland Chromatin', 'Normal Nucleoli', 'Mitoses']

input_data = {}

for feature in features:
    input_data[feature] = st.sidebar.selectbox(feature + ":", list(range(1, 11)))


# DataFrame untuk input user
input_df = pd.DataFrame(input_data, index=[0])

st.header("User Input as DataFrame")
st.write(input_df)

# Memuat model
model = MyModel("model/my_model.pkl")

# Tombol untuk prediksi
if st.button("**Predict**"):
    prediction = model.predict(input_df)
    
    if prediction[0] == 2:
        result = ":green[**Benign**]"
    elif prediction[0] == 4:
        result = ":purple[**Malignant**]"
    
    st.subheader("Prediction:")
    st.subheader(result)