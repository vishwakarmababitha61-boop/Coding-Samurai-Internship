import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# Download historical stock data
stock = yf.download("AAPL", start="2020-01-01", end="2025-01-01")

# Create dataset
data = pd.DataFrame()
data["Close"] = stock["Close"]
data["Target"] = data["Close"].shift(-1)

# Remove missing values
data = data.dropna()

# Features and target
X = data[["Close"]]
y = data["Target"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate model
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("Model trained successfully!")
print("MAE:", mae)
print("RMSE:", rmse)
print("R² Score:", r2)

# Plot actual vs predicted prices
plt.figure(figsize=(12, 6))

plt.plot(
    y_test.values,
    label="Actual Price"
)

plt.plot(
    predictions,
    label="Predicted Price"
)

plt.title("AAPL Stock Price: Actual vs Predicted")
plt.xlabel("Test Data Points")
plt.ylabel("Stock Price (USD)")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()