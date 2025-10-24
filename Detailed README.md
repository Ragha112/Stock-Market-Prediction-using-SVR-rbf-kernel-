# 📈 Stock Market Prediction using Support Vector Regression (SVR) with RBF Kernel

This project applies **Machine Learning techniques (SVR with RBF kernel)** to forecast the stock prices of **Apple (AAPL)** and **Microsoft (MSFT)** using **5 years of historical data**.  
It integrates **technical indicators** and **hyperparameter tuning** to enhance prediction accuracy and model generalization.

---

## 🧠 Objective
To build a regression model capable of predicting **future closing prices** of major tech stocks by capturing nonlinear market trends using **Support Vector Regression (SVR)**.

---

## 📊 Dataset
- **Source:** [Yahoo Finance](https://finance.yahoo.com) (via `yfinance` API)
- **Stocks Used:** AAPL (Apple) and MSFT (Microsoft)
- **Time Period:** 2018–2023 (5 years)
- **Frequency:** Daily closing prices

| Column | Description |
|--------|--------------|
| Date | Trading day |
| Open | Opening price |
| High | Highest price of the day |
| Low | Lowest price of the day |
| Close | Closing price |
| Volume | Total shares traded |

---

## ⚙️ Tools & Libraries
| Library | Purpose |
|----------|----------|
| **yFinance** | Fetch historical stock data |
| **Pandas, NumPy** | Data manipulation and preprocessing |
| **Matplotlib, Seaborn** | Visualization |
| **scikit-learn (SVR, GridSearchCV)** | Model training and tuning |
| **EMA (Exponential Moving Average)** | Trendline indicator for smoothing price series |

---

## 🧩 Methodology

### 1️⃣ Data Collection
Data was fetched using the `yfinance` library for both Apple and Microsoft tickers.

### 2️⃣ Feature Engineering
- Calculated **EMA-14 (Exponential Moving Average)** to smooth out short-term fluctuations.  
- Normalized data using `StandardScaler` for SVR training.

### 3️⃣ Model Selection
- Used **Support Vector Regression (SVR)** with **RBF kernel** to capture nonlinear price patterns.  
- The RBF kernel helps model complex relationships between past and current prices.

### 4️⃣ Hyperparameter Tuning
Applied **GridSearchCV** to optimize parameters:
```python
parameters = {
    'C': [1, 10, 100],
    'gamma': [0.01, 0.1, 1],
    'epsilon': [0.01, 0.1, 1]
}
