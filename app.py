import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Student Marks Predictor", page_icon="🎓")

st.title("🎓 Student Marks Predictor")
st.write("AI Based Student Performance Prediction")

hours = np.array([1,2,3,4,5,6,7,8,9]).reshape(-1,1)
marks = np.array([15,25,35,45,55,65,75,85,95])

model = LinearRegression()
model.fit(hours, marks)

st.subheader("Enter Your Study Details")

study_hours = st.slider(
    "Daily Study Hours",
    1, 12, 5
)

if st.button("Predict My Marks"):
    prediction = model.predict([[study_hours]])[0]
    prediction = np.clip(prediction, 0, 100)

    st.success("Prediction Completed!")
    st.metric("Predicted Marks", f"{prediction:.2f}/100")

    fig, ax = plt.subplots()
    ax.scatter(hours, marks, label="Training Data")
    ax.plot(hours, model.predict(hours), label="AI Prediction")
    ax.scatter([study_hours], [prediction],
               color="red", label="Your Prediction")

    ax.set_xlabel("Study Hours")
    ax.set_ylabel("Marks")
    ax.legend()
    ax.grid()

    st.pyplot(fig)

st.caption("First Year AIDS Project")