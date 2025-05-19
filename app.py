
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE

st.set_page_config(page_title="Prediksi Obesitas", layout="wide")
st.title("📊 Aplikasi Prediksi Obesitas")

uploaded_file = st.file_uploader("Upload Dataset Obesitas (CSV)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Data Awal")
    st.dataframe(df.head())

    st.subheader("📌 Informasi Dataset")
    buffer = []
    df.info(buf=buffer := [])
    st.text("\n".join(buffer))

    st.subheader("📈 Statistik Deskriptif")
    st.dataframe(df.describe(include='all'))

    # Preprocessing
    le = LabelEncoder()
    for col in df.select_dtypes(include='object').columns:
        df[col] = le.fit_transform(df[col])

    X = df.drop('NObeyesdad', axis=1)
    y = df['NObeyesdad']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    st.subheader("⚙️ Pilih Model")
    model_choice = st.selectbox("Model", ["Random Forest", "SVM", "KNN"])

    if st.button("Latih Model"):
        if model_choice == "Random Forest":
            model = RandomForestClassifier()
        elif model_choice == "SVM":
            model = SVC()
        elif model_choice == "KNN":
            model = KNeighborsClassifier()

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        st.subheader("✅ Evaluasi Model")
        st.text("Akurasi: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))
        st.text("\nClassification Report:\n" + classification_report(y_test, y_pred))

        st.subheader("🌀 Confusion Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", ax=ax)
        st.pyplot(fig)
else:
    st.info("Silakan upload dataset untuk memulai.")
