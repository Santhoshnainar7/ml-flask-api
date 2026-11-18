
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression

# Sample house data
X = np.array([
    [800, 2, 1],
    [1000, 2, 2],
    [1200, 3, 2],
    [1500, 3, 2],
    [1800, 3, 3],
    [2000, 4, 3],
    [2500, 4, 3],
    [3000, 5, 4]
])

# House prices in lakhs
y = np.array([35, 45, 55, 65, 80, 95, 115, 140])

# Train the model
model = LinearRegression()
model.fit(X, y)

# Save the trained model
joblib.dump(model, "house_model.pkl")

print("Model trained and saved successfully!")
