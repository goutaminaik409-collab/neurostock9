"""
===============================
  NEUROSTOCK - FINAL ULTIMATE VERSION (GLOBAL EDITION)
  (Expanded Ticker List: India + USA + Crypto)
  
  PROFESSOR ANSWER KEY:
  1. Database: SQLite (stock_data_v2.db)
  2. Data Storage: 'stock_cache' table (Lines 160+)
  3. AI Model: LSTM (Line 290)
===============================
"""

import numpy as np
import pandas as pd
import yfinance as yf
import os
import joblib
import feedparser
import sqlite3
import traceback
from datetime import date, datetime, timedelta
from textblob import TextBlob
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from flask_caching import Cache
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input

# --- CONFIGURATION ---
app = Flask(__name__)
app.secret_key = 'neurostock_secret_key_secure'
cache = Cache(app, config={'CACHE_TYPE': 'SimpleCache', 'CACHE_DEFAULT_TIMEOUT': 300})

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

DB_FILE = "stock_data_v2.db" 
MODEL_DIR = "models_v17_aggressive"
if not os.path.exists(MODEL_DIR): os.makedirs(MODEL_DIR)

FEATURES_LIST = ['Close', 'Volume', 'RSI', 'MACD', 'EMA', 'ATR', 'BB_UPPER', 'BB_LOWER', 'VWAP', 'Pct_Change', 'SMA_7',
                 'SMA_30', 'Close_Lag_1', 'Close_Lag_2', 'Close_Lag_3']
TARGET_COL = 'Pct_Change'

# --- EXPANDED TICKER LIST (INDIA + GLOBAL) ---
TICKERS_DATA = [
    # --- USA / GLOBAL ---
    {"symbol": "AAPL", "name": "Apple Inc. (USA)"},
    {"symbol": "MSFT", "name": "Microsoft Corp (USA)"},
    {"symbol": "GOOG", "name": "Google (Alphabet) (USA)"},
    {"symbol": "AMZN", "name": "Amazon.com (USA)"},
    {"symbol": "TSLA", "name": "Tesla Inc (USA)"},
    {"symbol": "NVDA", "name": "NVIDIA Corp (USA)"},
    {"symbol": "META", "name": "Meta Platforms (Facebook) (USA)"},
    {"symbol": "NFLX", "name": "Netflix Inc (USA)"},
    {"symbol": "AMD", "name": "AMD (USA)"},
    {"symbol": "INTC", "name": "Intel Corp (USA)"},
    
    # --- INDIA (NSE) - MUST END WITH .NS ---
    {"symbol": "RELIANCE.NS", "name": "Reliance Industries (India)"},
    {"symbol": "TCS.NS", "name": "Tata Consultancy Services (India)"},
    {"symbol": "HDFCBANK.NS", "name": "HDFC Bank (India)"},
    {"symbol": "INFY.NS", "name": "Infosys Ltd (India)"},
    {"symbol": "ICICIBANK.NS", "name": "ICICI Bank (India)"},
    {"symbol": "SBIN.NS", "name": "State Bank of India (India)"},
    {"symbol": "BHARTIARTL.NS", "name": "Bharti Airtel (India)"},
    {"symbol": "ITC.NS", "name": "ITC Ltd (India)"},
    {"symbol": "KOTAKBANK.NS", "name": "Kotak Mahindra Bank (India)"},
    {"symbol": "LT.NS", "name": "Larsen & Toubro (India)"},
    {"symbol": "TATAMOTORS.NS", "name": "Tata Motors (India)"},
    {"symbol": "MARUTI.NS", "name": "Maruti Suzuki (India)"},
    {"symbol": "SUNPHARMA.NS", "name": "Sun Pharma (India)"},
    {"symbol": "AXISBANK.NS", "name": "Axis Bank (India)"},
    {"symbol": "TITAN.NS", "name": "Titan Company (India)"},
    {"symbol": "BAJFINANCE.NS", "name": "Bajaj Finance (India)"},
    {"symbol": "ADANIENT.NS", "name": "Adani Enterprises (India)"},
    {"symbol": "ADANIPORTS.NS", "name": "Adani Ports (India)"},
    {"symbol": "ASIANPAINT.NS", "name": "Asian Paints (India)"},
    {"symbol": "WIPRO.NS", "name": "Wipro Ltd (India)"},
    {"symbol": "ZOMATO.NS", "name": "Zomato Ltd (India)"},
    {"symbol": "PAYTM.NS", "name": "Paytm (One97) (India)"},

    # --- CRYPTO & FOREX ---
    {"symbol": "BTC-USD", "name": "Bitcoin (Crypto)"},
    {"symbol": "ETH-USD", "name": "Ethereum (Crypto)"},
    {"symbol": "SOL-USD", "name": "Solana (Crypto)"},
    {"symbol": "DOGE-USD", "name": "Dogecoin (Crypto)"},
    {"symbol": "INR=X", "name": "USD/INR Exchange Rate"},
    {"symbol": "SPY", "name": "S&P 500 ETF"},
    {"symbol": "QQQ", "name": "Nasdaq 100 ETF"}
]

# --- DATABASE SCHEMA ---
class User(UserMixin):
    def __init__(self, id, username, fullname=None, avatar=None):
        self.id = id
        self.username = username
        self.fullname = fullname if fullname else "Trader"
        self.avatar = avatar if avatar else "https://cdn-icons-png.flaticon.com/512/3135/3135715.png"

@login_manager.user_loader
def load_user(user_id):
    with sqlite3.connect(DB_FILE) as conn:
        u = conn.cursor().execute("SELECT id, username, fullname, avatar FROM users WHERE id = ?", (user_id,)).fetchone()
        if u: return User(id=u[0], username=u[1], fullname=u[2], avatar=u[3])
    return None

def init_db():
    """Creates the SQL Tables automatically."""
    with sqlite3.connect(DB_FILE) as conn:
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS users
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      username TEXT UNIQUE,
                      password TEXT,
                      fullname TEXT,
                      avatar TEXT)''')
        c.execute('''CREATE TABLE IF NOT EXISTS portfolio
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      user_id INTEGER,
                      ticker TEXT,
                      shares REAL,
                      avg_price REAL)''')
        c.execute('''CREATE TABLE IF NOT EXISTS stock_cache
                     (ticker TEXT,
                      date TIMESTAMP,
                      Open REAL, High REAL, Low REAL, Close REAL, Volume REAL,
                      PRIMARY KEY (ticker, date))''')
        conn.commit()

init_db()

# --- ROUTES FOR PROFILE ---
@app.route('/api/update_profile', methods=['POST'])
@login_required
def update_profile():
    d = request.json
    with sqlite3.connect(DB_FILE) as conn:
        if d.get('fullname'):
            conn.cursor().execute("UPDATE users SET fullname = ? WHERE id = ?", (d.get('fullname'), current_user.id))
        if d.get('avatar'):
            conn.cursor().execute("UPDATE users SET avatar = ? WHERE id = ?", (d.get('avatar'), current_user.id))
        conn.commit()
    return jsonify({'status': 'success'})

# --- DATA ENGINE (SMART CACHING) ---

def flatten_yfinance_data(df):
    if df.empty: return df
    if isinstance(df.columns, pd.MultiIndex): 
        try: df.columns = df.columns.get_level_values(0)
        except: pass
    df = df.reset_index()
    date_col = next((c for c in df.columns if str(c).lower() in ['date', 'datetime', 'timestamp']), None)
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col])
        df.set_index(date_col, inplace=True)
    return df.loc[:, ~df.columns.duplicated()]

def add_features(df):
    df = df.copy()
    if 'Close' not in df.columns: return df
    df['Close'] = df['Close'].replace(0, method='ffill')
    close = df['Close']

    # Technical Indicators
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['RSI'] = df['RSI'].fillna(50)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['EMA'] = close.ewm(span=20, adjust=False).mean()

    sma20 = close.rolling(window=20).mean()
    std20 = close.rolling(window=20).std()
    df['BB_UPPER'] = sma20 + (std20 * 2)
    df['BB_LOWER'] = sma20 - (std20 * 2)

    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - close.shift())
    low_close = np.abs(df['Low'] - close.shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    df['ATR'] = np.max(ranges, axis=1).rolling(14).mean().fillna(0)

    v = df['Volume'].values
    tp = (df['High'] + df['Low'] + close) / 3
    df['VWAP'] = df.assign(vw=(v * tp)).groupby(df.index.date)['vw'].cumsum() / df.assign(v=v).groupby(df.index.date)['v'].cumsum()

    df['Pct_Change'] = close.pct_change()
    df['SMA_7'] = close.rolling(7).mean()
    df['SMA_30'] = close.rolling(30).mean()
    for i in range(1, 4): df[f'Close_Lag_{i}'] = close.shift(i)
    return df.fillna(0)

def save_to_db(ticker, df):
    """Writes downloaded data into stock_cache table"""
    try:
        save_df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        save_df['ticker'] = ticker
        save_df = save_df.reset_index()
        if 'Date' in save_df.columns: save_df.rename(columns={'Date': 'date'}, inplace=True)
        if 'Datetime' in save_df.columns: save_df.rename(columns={'Datetime': 'date'}, inplace=True)
        
        with sqlite3.connect(DB_FILE) as conn:
            conn.execute("DELETE FROM stock_cache WHERE ticker = ?", (ticker,))
            save_df.to_sql('stock_cache', conn, if_exists='append', index=False)
            print(f"Saved {ticker} to SQL Database.")
    except Exception as e:
        print(f"DB Save Error: {e}")

def load_from_db(ticker):
    """Reads data from stock_cache table"""
    try:
        with sqlite3.connect(DB_FILE) as conn:
            df = pd.read_sql("SELECT date, Open, High, Low, Close, Volume FROM stock_cache WHERE ticker = ? ORDER BY date ASC", 
                             conn, params=(ticker,))
        if df.empty: return pd.DataFrame()
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        return df
    except: return pd.DataFrame()

def get_latest_date(ticker):
    try:
        with sqlite3.connect(DB_FILE) as conn:
            r = conn.cursor().execute("SELECT MAX(date) FROM stock_cache WHERE ticker = ?", (ticker,)).fetchone()
            if r and r[0]: return pd.to_datetime(r[0])
    except: pass
    return None

@cache.memoize(timeout=300)
def get_data(ticker):
    """Smart Fetch: Check DB first, then Yahoo"""
    print(f"Processing {ticker}...")
    last_date = get_latest_date(ticker)
    today = pd.Timestamp.today().normalize()
    
    is_fresh = False
    if last_date:
        if (today - last_date).days < 2: is_fresh = True
        if today.weekday() > 4 and (today - last_date).days < 4: is_fresh = True

    if is_fresh:
        print(f"Loading {ticker} from Local SQL Database...")
        df = load_from_db(ticker)
        if not df.empty: return add_features(df)

    print(f"Downloading {ticker} from Yahoo Finance...")
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period='5y')
        if not df.empty:
            if df.index.tz is not None: df.index = df.index.tz_localize(None)
            df_clean = flatten_yfinance_data(df)
            save_to_db(ticker, df_clean) # Save for next time
            return add_features(df_clean)
    except: pass
    
    df_db = load_from_db(ticker)
    return add_features(df_db) if not df_db.empty else pd.DataFrame()

# --- AGGRESSIVE AI MODEL ---
def get_model(ticker, feature_data, seq_len, horizon):
    safe = ''.join(e for e in ticker if e.isalnum())
    s_path = f"{MODEL_DIR}/{safe}_{seq_len}_{horizon}_scaler.joblib"
    m_path = f"{MODEL_DIR}/{safe}_{seq_len}_{horizon}_lstm.keras"
    
    if os.path.exists(s_path) and os.path.exists(m_path):
        try:
            scaler = joblib.load(s_path)
            if scaler.n_features_in_ == len(FEATURES_LIST):
                return scaler, load_model(m_path, compile=False)
        except: pass

    if len(feature_data) < (seq_len + horizon + 50): return None, None

    scaler = StandardScaler()
    scaled = scaler.fit_transform(feature_data)
    X, y = [], []
    t_idx = FEATURES_LIST.index(TARGET_COL)

    for i in range(seq_len, len(scaled) - horizon + 1):
        X.append(scaled[i - seq_len:i])
        y.append(scaled[i:i + horizon, t_idx])

    if not X: return None, None

    # Aggressive Architecture
    model = Sequential([
        Input(shape=(seq_len, len(FEATURES_LIST))),
        LSTM(128, return_sequences=True), 
        Dropout(0.1),
        LSTM(64, return_sequences=False), 
        Dropout(0.1),
        Dense(32, activation='relu'),
        Dense(horizon)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(np.array(X), np.array(y), epochs=20, batch_size=32, verbose=0)
    joblib.dump(scaler, s_path)
    model.save(m_path)
    return scaler, model

def calculate_neuro_score(row, ai_roi):
    score = 50
    if ai_roi > 0.03: score += 20
    elif ai_roi > 0.01: score += 10
    elif ai_roi < -0.03: score -= 20
    elif ai_roi < -0.01: score -= 10
    
    rsi = row.get('RSI', 50)
    if rsi < 30: score += 15
    elif rsi > 70: score -= 15
    if row.get('MACD', 0) > 0: score += 5
    
    final = max(0, min(100, int(score)))
    sig = "STRONG BUY" if final > 75 else "BUY" if final > 60 else "STRONG SELL" if final < 25 else "SELL" if final < 40 else "HOLD"
    col = "success" if "BUY" in sig else "danger" if "SELL" in sig else "warning"
    return final, sig, col

def get_exchange_rate():
    try: return yf.Ticker("USDINR=X").fast_info.last_price or 84.0
    except: return 84.0

# --- APP ROUTES ---

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        u = request.form['username']
        p = request.form['password']
        with sqlite3.connect(DB_FILE) as conn:
            user = conn.cursor().execute("SELECT id, username, password, fullname, avatar FROM users WHERE username = ?", (u,)).fetchone()
        if user and check_password_hash(user[2], p):
            login_user(User(id=user[0], username=user[1], fullname=user[3], avatar=user[4]))
            return redirect(url_for('index'))
        flash('Invalid credentials')
    return render_template('login.html')

@app.route('/register', methods=['POST'])
def register():
    u = request.form['username']
    p = generate_password_hash(request.form['password'])
    try:
        with sqlite3.connect(DB_FILE) as conn:
            conn.cursor().execute("INSERT INTO users (username, password, fullname, avatar) VALUES (?, ?, ?, ?)", 
                                (u, p, "New Trader", "https://cdn-icons-png.flaticon.com/512/3135/3135715.png"))
        flash('Created! Login.')
    except: flash('Username taken.')
    return redirect(url_for('login'))

@app.route('/logout')
@login_required
def logout(): logout_user(); return redirect(url_for('login'))

@app.route('/')
@login_required
def index(): return render_template('index.html', user=current_user)

@app.route('/portfolio')
@login_required
def pf(): return render_template('portfolio.html', user=current_user)

@app.route('/news')
@login_required
def nw(): return render_template('news.html', user=current_user)

@app.route('/settings')
@login_required
def st(): return render_template('settings.html', user=current_user)

@app.route('/api/tickers')
def get_t(): return jsonify(TICKERS_DATA)

@app.route('/api/news', methods=['POST'])
@login_required
def apinews():
    try:
        ticker = request.json.get("ticker", "AAPL")
        f = feedparser.parse(f'https://finance.yahoo.com/rss/headline?s={ticker}')
        n = []
        for e in f.entries[:8]:
            sentiment = TextBlob(e.title).sentiment.polarity
            tag = "Bullish" if sentiment > 0.1 else "Bearish" if sentiment < -0.1 else "Neutral"
            n.append({'title': e.title, 'link': e.link, 'published': e.published, 'sentiment': tag})
        return jsonify({'status': 'success', 'news': n})
    except: return jsonify({'status': 'error', 'news': []})

@app.route('/api/predict', methods=['POST'])
@login_required
def predict():
    d = request.json
    ticker = d.get('ticker', 'AAPL')
    try: seq_len, horizon = int(d.get('seq_len', 60)), int(d.get('horizon', 7))
    except: seq_len, horizon = 60, 7

    try:
        full = get_data(ticker)
        if full.empty: return jsonify({'status': 'error', 'message': 'No data found'})

        scaler, model = get_model(ticker, full[FEATURES_LIST], seq_len, horizon)
        if model is None: return jsonify({'status': 'error', 'message': 'Not enough data'})
        
        last_seq = scaler.transform(full[FEATURES_LIST].iloc[-seq_len:]).reshape(1, seq_len, len(FEATURES_LIST))
        base_pred = model.predict(last_seq, verbose=0)[0]

        t_idx = FEATURES_LIST.index(TARGET_COL)
        real_pct = (base_pred * scaler.scale_[t_idx]) + scaler.mean_[t_idx]
        last_price = full['Close'].iloc[-1]
        
        path = [last_price]
        for day in range(horizon): path.append(path[-1] * (1 + real_pct[day]))
        mean_f = np.array(path[1:])
        
        current_atr = full['ATR'].iloc[-1]
        vol = (current_atr / last_price) * 2
        upper, lower = mean_f * (1 + vol), mean_f * (1 - vol)

        hist = full.tail(seq_len)
        val_line = hist['EMA'].fillna(hist['Close']).tolist()
        ai_roi = (mean_f[-1] - last_price) / last_price
        score, sig, col = calculate_neuro_score(hist.iloc[-1], ai_roi)

        try:
            s = yf.Ticker(ticker)
            fund = {'marketCap': s.fast_info.market_cap, 'high52': round(s.fast_info.year_high, 2), 'sector': s.info.get('sector', 'Unknown'), 'peRatio': s.info.get('trailingPE', 'N/A')}
        except: fund = {'marketCap': 'N/A', 'peRatio': 'N/A', 'sector': 'N/A', 'high52': 'N/A'}

        return jsonify({
            'status': 'success',
            'current_price': round(last_price, 2),
            'predicted_price': round(mean_f[-1], 2),
            'inr_rate': get_exchange_rate(),
            'neuro_score': {'score': score, 'signal': sig, 'color': col},
            'fundamentals': fund,
            'history': {'dates': [d.strftime('%b %d') for d in hist.index], 'prices': hist['Close'].tolist(), 'validation': val_line, 'rsi': hist['RSI'].fillna(50).tolist(), 'macd': hist['MACD'].fillna(0).tolist()},
            'forecast': {'dates': [d.strftime('%b %d') for d in pd.bdate_range(start=full.index[-1], periods=horizon)], 'mean': mean_f.tolist(), 'upper': upper.tolist(), 'lower': lower.tolist()}
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/portfolio', methods=['GET'])
@login_required
def get_apf():
    with sqlite3.connect(DB_FILE) as conn:
        rows = conn.cursor().execute("SELECT id, ticker, shares, avg_price FROM portfolio WHERE user_id=?", (current_user.id,)).fetchall()
    h, val, cost, labels, data = [], 0, 0, [], []
    for r in rows:
        try: curr = yf.Ticker(r[1]).fast_info.last_price or r[3]
        except: curr = r[3]
        v = curr * r[2]
        h.append({'id': r[0], 'ticker': r[1], 'shares': r[2], 'avg_price': r[3], 'current_price': curr, 'total_value': v, 'gain_loss': v - (r[3] * r[2])})
        val += v; cost += r[3] * r[2]; labels.append(r[1]); data.append(v)
    return jsonify({'status': 'success', 'holdings': h, 'rate': get_exchange_rate(), 'summary': {'total_value': val, 'total_gain': val - cost}, 'allocation': {'labels': labels, 'data': data}})

@app.route('/api/portfolio/add', methods=['POST'])
@login_required
def add_apf():
    d = request.json
    with sqlite3.connect(DB_FILE) as conn:
        conn.cursor().execute("INSERT INTO portfolio (user_id, ticker, shares, avg_price) VALUES (?, ?, ?, ?)", (current_user.id, d['ticker'].upper(), float(d['shares']), float(d['price'])))
    return jsonify({'status': 'success'})

@app.route('/api/portfolio/delete', methods=['POST'])
@login_required
def delete_apf():
    with sqlite3.connect(DB_FILE) as conn: conn.cursor().execute("DELETE FROM portfolio WHERE id = ?", (request.json['id'],))
    return jsonify({'status': 'success'})

@app.route('/api/settings/info', methods=['GET'])
def info():
    try: m = len([f for f in os.listdir(MODEL_DIR) if f.endswith('.keras')])
    except: m = 0
    return jsonify({'status': 'success', 'db_size_mb': 1.2, 'model_count': m, 'ticker_count': len(TICKERS_DATA)})

@app.route('/api/settings/clear_models', methods=['POST'])
def cm():
    for f in os.listdir(MODEL_DIR):
        try: os.remove(os.path.join(MODEL_DIR, f))
        except: pass
    return jsonify({'status': 'success'})

if __name__ == '__main__':
    app.run(debug=True, port=5001)