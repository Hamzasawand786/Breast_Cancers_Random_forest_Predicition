# app.py
import streamlit as st
import pickle
import numpy as np

class IrisPredictionApp:
    def __init__(self, model_path: str):
        self.model = self.load_model(model_path)
        self.classes = {
            0: "Iris Setosa",
            1: "Iris Versicolor",
            2: "Iris Virginica"
        }

    def load_model(self, path):
        with open(path, "rb") as f:
            return pickle.load(f)

    def render_header(self):
        st.set_page_config(page_title="Iris Prediction", layout="centered")
        st.title("🌸 Iris Flower Prediction")
        st.markdown("Predict Iris species using trained ML model.")

    def render_inputs(self):
        with st.container():
            st.subheader("Input Features")
            col1, col2 = st.columns(2)

            with col1:
                self.sepal_length = st.number_input(
                    "Sepal Length (cm)", min_value=0.0, format="%.2f"
                )
                self.sepal_width = st.number_input(
                    "Sepal Width (cm)", min_value=0.0, format="%.2f"
                )

            with col2:
                self.petal_length = st.number_input(
                    "Petal Length (cm)", min_value=0.0, format="%.2f"
                )
                self.petal_width = st.number_input(
                    "Petal Width (cm)", min_value=0.0, format="%.2f"
                )

    def predict(self):
        input_data = np.array([[
            self.sepal_length,
            self.sepal_width,
            self.petal_length,
            self.petal_width
        ]])
        prediction = self.model.predict(input_data)[0]
        return prediction

    def render_prediction(self):
        if st.button("Predict"):
            pred = self.predict()
            class_name = self.classes.get(pred, str(pred))
            st.success(f"Predicted Class: {class_name}")

    def render_footer(self):
        st.markdown("---")
        st.caption("Streamlit ML App")

    def run(self):
        self.render_header()
        self.render_inputs()
        self.render_prediction()
        self.render_footer()


if __name__ == "__main__":
    app = IrisPredictionApp("breast_cancer_random_forest.pkl")
    app.run()
