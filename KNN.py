import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# Load dataset
data_train = pd.read_hdf('iris_classification.h5', key='fitur_training')
kelas_train = pd.read_hdf('iris_classification.h5', key='label_training')

# Train model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(data_train, kelas_train)

# Streamlit app
st.set_page_config(page_title="KNN Iris Classifier", page_icon="🌸")

st.title("KNN Iris Classifier")
st.markdown("Masukkan nilai fitur bunga iris untuk memprediksi kelasnya menggunakan algoritma KNN.")

# Form input
with st.form(key="iris_form"):
    st.subheader("Input Nilai Fitur")
    SepalLengthCm = st.number_input("Sepal Length (cm)", min_value=0.0, step=0.1)
    SepalWidthCm = st.number_input("Sepal Width (cm)", min_value=0.0, step=0.1)
    PetalLengthCm = st.number_input("Petal Length (cm)", min_value=0.0, step=0.1)
    PetalWidthCm = st.number_input("Petal Width (cm)", min_value=0.0, step=0.1)
    submit_button = st.form_submit_button(label="Prediksi")

# Handle form submission
if submit_button:
    if not all([SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm]):
        st.error("Mohon masukkan semua nilai untuk fitur-fitur yang dibutuhkan.")
    else:
        # Convert input into numpy array and reshape
        test_data = np.array([SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm])
        test_data = test_data.reshape(1, -1)

        # Predict using KNN
        hasil = knn.predict(test_data)
        st.success(f"Hasil prediksi k-NN: **{hasil[0]}**")
