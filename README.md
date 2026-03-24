# 🥦 Item Price Trend & LSTM Forecasting App

This project is a **Streamlit web application** that predicts and visualizes future item prices using a trained **LSTM (Long Short-Term Memory) deep learning model**. It analyzes historical price data and generates both **test predictions and future forecasts** for selected items.

---

## 📌 Features

- 📊 Interactive Streamlit dashboard
- 📦 Select any item from dataset
- 📈 Visualize:
  - Actual price trends
  - Model predictions (test data)
  - Future price forecasts
- 🤖 LSTM-based time series forecasting
- 🔄 Scaled data preprocessing using saved scaler
- 📅 Future prediction timeline generation
- 📉 Plotly interactive charts

---

## 🧠 Model Overview

The forecasting system uses:

- **LSTM Neural Network (Keras)**
- **MinMax/Standard Scaler (pickle saved)**
- Sliding window sequence generation
- Time-series forecasting approach

### Workflow:
1. Load historical price dataset
2. Preprocess and scale data
3. Create time sequences (window-based)
4. Train/test split on sequences
5. Predict test data
6. Forecast future values iteratively

---

## 📊 Dataset

The dataset is loaded from a Google Sheets CSV export:

- Columns used:
  - `friendly_name`
  - `date`
  - `average_price`
  - `price_range_min`
  - `price_range_max`

---

## 🚀 How to Run the Project

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Streamlit app
```bash
streamlit run app.py
```
