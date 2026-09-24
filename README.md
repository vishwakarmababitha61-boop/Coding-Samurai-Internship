# AI Stock Price Prediction

## 📌 Project Overview

This project uses Machine Learning to predict the next day's Apple (AAPL) stock closing price based on historical stock price data.

The project uses Linear Regression and evaluates the model using MAE, RMSE, and R² Score.

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- yFinance

## 📊 Dataset

Historical AAPL stock data is downloaded using the Yahoo Finance API through the `yfinance` Python library.

Data period:
- 2020-01-01 to 2025-01-01

## 🤖 Machine Learning Model

**Linear Regression**

The model uses the previous closing price to predict the next day's closing price.

## 📈 Model Results

- MAE: 2.15
- RMSE: 2.90
- R² Score: 0.987

## 📉 Visualization

The project generates a graph comparing the actual stock prices with the predicted stock prices.

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone <YOUR-GITHUB-REPOSITORY-URL>