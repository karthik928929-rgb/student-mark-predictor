from sklearn.linear_model import LinearRegression
import numpy as np

# Training data
hours = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]).reshape(-1, 1)
marks = np.array([15, 25, 35, 45, 55, 65, 75, 85, 95])

# Create model
model = LinearRegression()
model.fit(hours, marks)

# Get user input
study_hours = float(input("Enter study hours: "))

# Predict marks
prediction = model.predict([[study_hours]])

print("Predicted Marks:", round(prediction[0], 2))