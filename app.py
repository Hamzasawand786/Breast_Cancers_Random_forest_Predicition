# app.py
import streamlit as st
import pickle
import joblib
import numpy as np

# Page config must be first
st.set_page_config(page_title="Iris Prediction", layout="centered")


class IrisPredictionApp:
    def __init__(self, model_path: str):
        self.model = self.load_model(model_path)
        self.classes = {
            0: "Iris Setosa",
            1: "Iris Versicolor",
            2: "Iris Virginica"
        }

    def load_model(self, path):
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            return joblib.load(path)

    def render_header(self):
        st.title("🌸 Iris Flower Prediction")

    def render_inputs(self):
        st.subheader("Input Features")

        col1, col2 = st.columns(2)

        with col1:
            self.sepal_length = st.number_input("Sepal Length (cm)", value=5.1)
            self.sepal_width = st.number_input("Sepal Width (cm)", value=3.5)

        with col2:
            self.petal_length = st.number_input("Petal Length (cm)", value=1.4)
            self.petal_width = st.number_input("Petal Width (cm)", value=0.2)

    def render_prediction(self):
        if st.button("Predict"):
            input_data = np.array([[
                self.sepal_length,
                self.sepal_width,
                self.petal_length,
                self.petal_width
            ]])

            prediction = self.model.predict(input_data)[0]
            st.success(f"Predicted Class: {self.classes.get(prediction, prediction)}")

    def run(self):
        self.render_header()
        self.render_inputs()
        self.render_prediction()


if __name__ == "__main__":
    app = IrisPredictionApp("breast_cancer_random_forest.pkl")
    app.run()
