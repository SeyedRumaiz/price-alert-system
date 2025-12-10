import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from keras.models import load_model
import pickle

# ---------------------------
# 1. Load Data
# ---------------------------
@st.cache_data
def load_data():
    url = "https://docs.google.com/spreadsheets/d/1yTAEczMeq9kXkAYKqhPkH9wXd__xYqk9RVK-GKjqNSY/export?format=csv&gid=0"
    df = pd.read_csv(url)

    # Keep only necessary columns
    df = df[['friendly_name', 'date', 'average_price', "price_range_min", "price_range_max"]].copy()

    # Convert date
    df['date'] = pd.to_datetime(df['date'])

    # Drop rows with missing values
    df.dropna(inplace=True)

    # Sort
    df.sort_values(['friendly_name', 'date'], inplace=True)

    return df

# ---------------------------
# 2. Load model and scaler
# ---------------------------
@st.cache_resource
def load_lstm_model():
    model = load_model("model.keras")
    return model

@st.cache_resource
def load_scaler():
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return scaler

# ---------------------------
# 3. Create sequences
# ---------------------------
def create_sequences(data, window=24):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i + window])
        y.append(data[i + window])
    return np.array(X), np.array(y)

# ---------------------------
# 4. Predict function
# ---------------------------
def predict_prices(df_product, model, scaler, window=12, future_steps=10):
    df_product = df_product.copy()
    prices_scaled = scaler.transform(df_product[['average_price']].values)

    # Create sequences
    X, y = create_sequences(prices_scaled, window)

    # Split last 20% as test
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Predict test
    y_pred_scaled = model.predict(X_test)
    y_true = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred = scaler.inverse_transform(y_pred_scaled).flatten()

    # Predict future
    last_sequence = prices_scaled[-window:]
    future_preds = []
    seq = last_sequence.copy()

    for _ in range(future_steps):
        pred = model.predict(seq.reshape(1, window, 1))[0][0]
        future_preds.append(pred)
        seq = np.append(seq[1:], [[pred]], axis=0)

    future_preds = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1)).flatten()

    return y_true, y_pred, future_preds

# ---------------------------
# 5. Streamlit App
# ---------------------------
def main():
    st.title("🥦 Item Price Trend + LSTM Forecast")

    df = load_data()
    model = load_lstm_model()
    scaler = load_scaler()

    # Dropdown only shows items in the filtered df
    product_list = sorted(df['friendly_name'].unique())
    product = st.selectbox("Select an item:", product_list)

    if product:
        df_product = df[df['friendly_name'] == product]

        if len(df_product) < 20:
            st.warning("Not enough data to make predictions.")
            return

        # Predict
        y_true, y_pred, future_preds = predict_prices(df_product, model, scaler, window=12, future_steps=10)

        # Dates
        actual_dates = df_product['date'].iloc[-len(y_true):]
        future_dates = pd.date_range(
            start=df_product['date'].iloc[-1],
            periods=len(future_preds) + 1,
            freq="7D"
        )[1:]

        # Plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=actual_dates,
            y=y_true,
            mode='lines',
            name='Actual Price',
            line=dict(color='green')
        ))
        fig.add_trace(go.Scatter(
            x=actual_dates,
            y=y_pred,
            mode='lines',
            name='Predicted Price',
            line=dict(color='red', dash='dash')
        ))
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=future_preds,
            mode='lines+markers',
            name='Future Forecast',
            line=dict(color='blue', width=3)
        ))

        fig.update_layout(
            title=f"LSTM Price Forecast for {product}",
            xaxis_title="Date",
            yaxis_title="Price (LKR)",
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Future forecast table
        st.subheader("🔮 Future Forecasted Prices")
        forecast_df = pd.DataFrame({
            "Date": future_dates,
            "Predicted Price (LKR)": future_preds
        })
        st.table(forecast_df)

if __name__ == "__main__":
    main()
