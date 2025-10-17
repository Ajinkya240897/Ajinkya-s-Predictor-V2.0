# ajinkya_predictor_v2_ready.py
# Ajinkya's Predictor V2.0 — Ready-to-paste improved app
# - UI unchanged (Ticker, FMP API key optional, Interval)
# - Enhanced math, robust fetchers, better ensemble, fixed-value issues fixed
# - No internal diagnostics shown on UI
# Requirements (minimum): streamlit, pandas, numpy
# Recommended: yfinance, statsmodels, vaderSentiment, altair
# Run: pip install -r requirements.txt && streamlit run ajinkya_predictor_v2_ready.py

import streamlit as st
import pandas as pd
import numpy as np
import math, requests, warnings
from datetime import datetime, timedelta
from functools import lru_cache

warnings.filterwarnings("ignore")

# Optional libs
try:
    import yfinance as yf
    HAVE_YFINANCE = True
except Exception:
    HAVE_YFINANCE = False

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    HAVE_STATSMODELS = True
except Exception:
    HAVE_STATSMODELS = False

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except Exception:
    VADER_AVAILABLE = False

st.set_page_config(page_title="Ajinkya's Predictor V2.0", layout="wide")
st.markdown("""
<div style="background:linear-gradient(90deg,#082f22,#0b6e4f); padding:16px; border-radius:12px; color: white; display:flex;align-items:center; gap:14px;">
  <div style="width:56px;height:56px;background:linear-gradient(135deg,#66cdaa,#0b6e4f);border-radius:10px;display:flex;align-items:center;justify-content:center;font-weight:800;font-size:20px;color:#042f1b;">
    AJ
  </div>
  <div>
    <div style="font-size:28px; font-weight:800; color:white; letter-spacing:0.4px;">Ajinkya's Predictor V2.0</div>
    <div style="font-size:12px; color:#dff6ea; margin-top:4px;">Improved — advanced math, robust fetching, better ensemble</div>
  </div>
</div>
""", unsafe_allow_html=True)

# Inputs
with st.form("inputs", clear_on_submit=False):
    left, right = st.columns([3,1])
    with left:
        fmp_key = st.text_input("FMP API Key (optional)", type="password")
        ticker_raw = st.text_input("Ticker (Indian, e.g., TCS, RELIANCE)", value="TCS")
    with right:
        interval = st.selectbox("Interval", ["3d","15d","1m","3m","6m","1y"])
        submitted = st.form_submit_button("Run Prediction")

MAP = {"3d":3, "15d":15, "1m":22, "3m":66, "6m":132, "1y":252}

def append_nse_if_needed(ticker: str):
    t = ticker.strip().upper()
    if "." in t: return t
    return t + ".NS"

# ---------- HTTP helper ----------
def safe_get(url, params=None, timeout=8):
    try:
        r = requests.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None

# ---------- Price history fetch ----------
@lru_cache(maxsize=64)
def fetch_prices_yf(ticker: str, days=1500):
    if not HAVE_YFINANCE:
        return None
    try:
        tk = yf.Ticker(ticker)
        end = datetime.now()
        start = end - timedelta(days=int(days*1.1))
        df = tk.history(start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"), interval="1d", actions=False)
        if df is None or df.empty:
            return None
        df = df.reset_index().rename(columns={"Date":"date","Close":"close","High":"high","Low":"low","Volume":"volume"})
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        return df[["date","close","high","low","volume"]]
    except Exception:
        return None

# ---------- Current price fetcher (FMP -> Yahoo intraday/fast_info -> daily close fallback) ----------
@lru_cache(maxsize=128)
def try_fmp_quote(ticker_short: str, fmp_key: str):
    if not fmp_key:
        return None
    try:
        for tk in (ticker_short, ticker_short + ".NS"):
            url = f"https://financialmodelingprep.com/api/v3/quote/{tk}"
            data = safe_get(url, params={"apikey": fmp_key})
            if isinstance(data, list) and len(data) > 0:
                item = data[0]
                price = item.get("price") or item.get("close") or item.get("previousClose")
                if price is not None:
                    try:
                        return float(price)
                    except Exception:
                        continue
    except Exception:
        pass
    return None

@lru_cache(maxsize=128)
def try_yf_current(ticker_ns: str):
    if not HAVE_YFINANCE:
        return None
    try:
        tk = yf.Ticker(ticker_ns)
        fast = getattr(tk, "fast_info", None)
        if isinstance(fast, dict):
            for k in ("last_price","lastPrice","last_trade_price","last"):
                if k in fast and fast[k] is not None:
                    try:
                        return float(fast[k])
                    except Exception:
                        pass
        try:
            hist = tk.history(period="2d", interval="1m", actions=False)
            if hist is not None and len(hist) > 0:
                last = hist["Close"].iloc[-1]
                if not pd.isna(last):
                    return float(last)
        except Exception:
            pass
        try:
            histd = tk.history(period="7d", interval="1d", actions=False)
            if histd is not None and len(histd) > 0:
                last = histd["Close"].iloc[-1]
                if not pd.isna(last):
                    return float(last)
        except Exception:
            pass
    except Exception:
        pass
    return None

@lru_cache(maxsize=128)
def get_current_price_and_info(ticker_input: str, fmp_key: str = None):
    if not ticker_input:
        return None
    t_input = ticker_input.strip().upper()
    t_short = t_input.replace(".NS", "")
    t_ns = t_short + ".NS" if ".NS" not in t_input else t_input

    if fmp_key:
        p = try_fmp_quote(t_short, fmp_key)
        if p is not None:
            return p
        p = try_fmp_quote(t_short + ".NS", fmp_key)
        if p is not None:
            return p

    p = try_yf_current(t_ns)
    if p is not None:
        return p

    return None

# -------------------------
# Basic math & indicators
# -------------------------
def ema(series, span): 
    return series.ewm(span=span, adjust=False).mean()

def sma(series, window):
    return series.rolling(window=window, min_periods=1).mean()

def log_return(series):
    return np.log(series/series.shift(1)).replace([np.inf,-np.inf],0).fillna(0)

def zscore(series, window=20):
    m = series.rolling(window=window, min_periods=1).mean()
    s = series.rolling(window=window, min_periods=1).std().replace(0, np.nan)
    return ((series - m) / s).fillna(0)

def rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0).fillna(0)
    down = -1 * delta.clip(upper=0).fillna(0)
    ma_up = up.ewm(alpha=1.0/period, adjust=False).mean()
    ma_down = down.ewm(alpha=1.0/period, adjust=False).mean()
    rs = ma_up / ma_down.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def macd_components(series, n_fast=12, n_slow=26, n_sig=9):
    fast = series.ewm(span=n_fast, adjust=False).mean()
    slow = series.ewm(span=n_slow, adjust=False).mean()
    macd_line = fast - slow
    signal = macd_line.ewm(span=n_sig, adjust=False).mean()
    hist = macd_line - signal
    return macd_line, signal, hist

def obv(df):
    try:
        obv_series = [0]
        for i in range(1, len(df)):
            if df["close"].iloc[i] > df["close"].iloc[i-1]:
                obv_series.append(obv_series[-1] + df["volume"].iloc[i])
            elif df["close"].iloc[i] < df["close"].iloc[i-1]:
                obv_series.append(obv_series[-1] - df["volume"].iloc[i])
            else:
                obv_series.append(obv_series[-1])
        return pd.Series(obv_series, index=df.index)
    except Exception:
        return pd.Series(np.zeros(len(df)), index=df.index)

def stochastic_oscillator(df, k_window=14, d_window=3):
    low_min = df["low"].rolling(k_window).min()
    high_max = df["high"].rolling(k_window).max()
    k = 100 * ((df["close"] - low_min) / (high_max - low_min + 1e-9))
    d = k.rolling(d_window).mean()
    return k.fillna(50), d.fillna(50)

def hurst_exponent(ts, lags_range=range(2, 50)):
    ts = np.array(ts.dropna())
    if len(ts) < 100:
        return 0.5
    rs = []
    for lag in lags_range:
        pp = np.subtract(ts[lag:], ts[:-lag])
        rs.append(np.std(pp))
    rs = np.array(rs)
    with np.errstate(divide='ignore', invalid='ignore'):
        poly = np.polyfit(np.log(lags_range), np.log(rs + 1e-9), 1)
    return max(0.0, min(1.0, poly[0]))

def half_life(series):
    try:
        series = series.dropna()
        if len(series) < 30:
            return np.nan
        x = np.log(series)
        phi = np.corrcoef(x.values[:-1], x.values[1:])[0,1]
        if abs(phi) >= 1:
            return np.inf
        return float(-np.log(2) / np.log(abs(phi))) if phi != 0 else np.inf
    except Exception:
        return np.nan

def kalman_smooth(series):
    try:
        n = len(series); xhat = np.zeros(n); P = np.zeros(n)
        Q = 1e-5; R = np.var(series) * 0.01 + 1e-6
        xhat[0] = series.iloc[0]; P[0] = 1.0
        for k in range(1, n):
            xhatminus = xhat[k-1]; Pminus = P[k-1] + Q
            K = Pminus / (Pminus + R)
            xhat[k] = xhatminus + K * (series.iloc[k] - xhatminus)
            P[k] = (1 - K) * Pminus
        return pd.Series(xhat, index=series.index)
    except Exception:
        return series

def ssa_trend_and_forecast(series, window=60, components=3, days_ahead=3):
    try:
        x = np.array(series.dropna())[-window:]
        n = len(x)
        if n < 2 * components:
            return float(series.iloc[-1]), float(series.iloc[-1])
        L = max(2, n // 2)
        K = n - L + 1
        X = np.column_stack([x[i:i+K] for i in range(L)])
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        Xr = sum(s[i] * np.outer(U[:,i], Vt[i,:]) for i in range(min(components, len(s))))
        recon = np.zeros(n)
        counts = np.zeros(n)
        for i in range(L):
            for j in range(K):
                recon[i+j] += Xr[i, j]
                counts[i+j] += 1
        recon = recon / (counts + 1e-9)
        t = np.arange(len(recon))
        p = np.polyfit(t, recon, 1)
        forecast = np.polyval(p, len(recon) + days_ahead - 1)
        return float(recon[-1]), float(forecast)
    except Exception:
        return float(series.iloc[-1]), float(series.iloc[-1])

def fft_dominant_forecast(series, days_ahead=3, top_k=4):
    x = series.values; n = len(x)
    if n < 12: return float(series.iloc[-1])
    t = np.arange(n)
    p = np.polyfit(t, x, 1); trend = np.polyval(p, t); resid = x - trend
    fft = np.fft.rfft(resid); freqs = np.fft.rfftfreq(n); amps = np.abs(fft)
    idx = np.argsort(amps)[-top_k:]
    future_t = np.arange(n, n + days_ahead); recon = np.zeros(days_ahead)
    for i in idx:
        a = fft[i]; freq = freqs[i]; phase = np.angle(a); amplitude = np.abs(a) / n * 2
        recon += amplitude * np.cos(2 * np.pi * freq * future_t + phase)
    last_trend = np.polyval(p, n - 1); slope = p[0]
    trend_fore = last_trend + slope * np.arange(1, days_ahead + 1)
    forecast = trend_fore + recon
    return float(forecast[-1])

# ---------- Advanced math helpers ----------
def atr(df, window=14):
    h = df["high"]; l = df["low"]; c = df["close"]
    tr1 = h - l
    tr2 = (h - c.shift(1)).abs()
    tr3 = (l - c.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window).mean().fillna(method="bfill")

def bollinger_bands(series, window=20, n_std=2):
    ma = series.rolling(window).mean()
    std = series.rolling(window).std()
    upper = ma + n_std * std
    lower = ma - n_std * std
    return ma.fillna(method="bfill"), upper.fillna(method="bfill"), lower.fillna(method="bfill")

def keltner_channel(df, ema_window=20, atr_window=10, mult=1.5):
    ema20 = df["close"].ewm(span=ema_window, adjust=False).mean()
    atrv = atr(df, window=atr_window)
    upper = ema20 + mult * atrv; lower = ema20 - mult * atrv
    return ema20.fillna(method="bfill"), upper.fillna(method="bfill"), lower.fillna(method="bfill")

def realized_volatility(returns, window=21):
    try:
        rv = returns.rolling(window).apply(lambda x: np.sqrt(np.sum(x**2) * (252/len(x))) if len(x)>0 else 0).fillna(0)
        return rv
    except Exception:
        return returns.rolling(window).std().fillna(0)

def shannon_entropy(series, bins=20):
    try:
        hist, edges = np.histogram(series.dropna(), bins=bins, density=True)
        probs = hist / (hist.sum() + 1e-9)
        probs = probs[probs>0]
        ent = -np.sum(probs * np.log(probs + 1e-12))
        return float(ent)
    except Exception:
        return 0.0

def sample_skewness(series):
    x = np.array(series.dropna())
    n = len(x)
    if n < 3: return 0.0
    m = x.mean(); s = x.std(ddof=1)
    return float((n/((n-1)*(n-2))) * np.sum(((x-m)/s)**3))

def sample_kurtosis(series):
    x = np.array(series.dropna())
    n = len(x)
    if n < 4: return 0.0
    m = x.mean(); s = x.std(ddof=1)
    g2 = (n*(n+1))/((n-1)*(n-2)*(n-3)) * np.sum(((x-m)/s)**4) - 3*(n-1)**2/((n-2)*(n-3))
    return float(g2)

# ---------- Feature preparation ----------
def prepare_features(df):
    df = df.copy()
    df["close"] = df["close"].astype(float)
    df["return1"] = df["close"].pct_change().fillna(0)
    df["logret"] = log_return(df["close"])
    df["ema8"] = ema(df["close"], 8); df["ema21"] = ema(df["close"], 21); df["sma50"] = sma(df["close"], 50)
    df["vol20"] = df["return1"].rolling(20).std().fillna(0)
    df["ewma_vol"] = df["return1"].ewm(span=20, adjust=False).std().fillna(0)
    df["rsi14"] = rsi(df["close"]); df["ret_z"] = zscore(df["return1"])
    df["macd"], df["macd_sig"], df["macd_hist"] = macd_components(df["close"])
    df["obv"] = obv(df)
    df["sto_k"], df["sto_d"] = stochastic_oscillator(df)
    df["atr14"] = atr(df, window=14)
    df["bb_mid"], df["bb_upper"], df["bb_lower"] = bollinger_bands(df["close"], window=20, n_std=2)
    df["kelt_mid"], df["kelt_upper"], df["kelt_lower"] = keltner_channel(df, ema_window=20, atr_window=10, mult=1.5)
    df["rv21"] = realized_volatility(df["return1"], window=21)
    df["entropy_30"] = df["close"].rolling(30).apply(lambda x: shannon_entropy(pd.Series(x)), raw=False).fillna(0)
    df["skew30"] = df["return1"].rolling(30).apply(lambda x: sample_skewness(pd.Series(x)), raw=False).fillna(0)
    df["kurt30"] = df["return1"].rolling(30).apply(lambda x: sample_kurtosis(pd.Series(x)), raw=False).fillna(0)
    for lag in [1,2,3,5,8,13,21]:
        df[f"lag_{lag}"] = df["close"].shift(lag)
    df["mom_5"] = df["close"].pct_change(5)
    df = df.dropna().reset_index(drop=True)
    return df

# ---------- Forecast building blocks ----------
def hw_predict(df, days_ahead=3):
    try:
        if not HAVE_STATSMODELS:
            raise Exception("statsmodels not available")
        hw = ExponentialSmoothing(df["close"], trend="add", seasonal=None, damped_trend=True)
        res = hw.fit(optimized=True)
        return float(res.forecast(days_ahead).iloc[-1])
    except Exception:
        return float(df["close"].iloc[-1])

def ar1_predict(df, days_ahead=3):
    try:
        r = df["logret"].dropna()
        if len(r) < 10:
            return float(df["close"].iloc[-1])
        phi = np.corrcoef(r[:-1], r[1:])[0,1] if len(r)>1 else 0
        mu = r.mean(); last = r.iloc[-1]
        forecast_logret = mu + (phi*(last - mu))
        price_fore = float(df["close"].iloc[-1] * math.exp(forecast_logret * days_ahead))
        return price_fore
    except Exception:
        return float(df["close"].iloc[-1])

def fft_predict(df, days_ahead=3):
    try:
        return float(fft_dominant_forecast(df["close"], days_ahead=days_ahead, top_k=6))
    except Exception:
        return float(df["close"].iloc[-1])

def trend_extrapolate(df, days_ahead=3):
    try:
        slope = df["close"].rolling(30).apply(lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x)>=3 else 0).iloc[-1]
        return float(df["close"].iloc[-1] + slope * days_ahead)
    except Exception:
        return float(df["close"].iloc[-1])

# ---------- Risk controls ----------
def historical_var(returns, alpha=0.05):
    try:
        returns = np.asarray(returns)
        if len(returns) < 10:
            return 0.0
        q = np.percentile(returns, alpha * 100)
        return float(max(0.0, -q))
    except Exception:
        return 0.0

def cvar_expected_shortfall(returns, alpha=0.05):
    try:
        returns = np.asarray(returns)
        if len(returns) < 10:
            return 0.0
        threshold = np.quantile(returns, alpha)
        tail = returns[returns <= threshold]
        if len(tail) == 0:
            return 0.0
        return float(max(0.0, -np.mean(tail)))
    except Exception:
        return 0.0

# ---------- Model evaluation helpers for dynamic ensemble ----------
def model_recent_stats(df, model_fn, days_ahead=3, lookback=90):
    n = len(df)
    if n < 40:
        return {"acc":0.5, "mape":np.inf}
    if n < lookback + days_ahead + 5:
        lookback = max(30, n // 3)
    start = n - lookback - days_ahead
    correct = 0; total = 0; errors = []
    for i in range(start, n - days_ahead):
        window = df.iloc[:i+1]
        try:
            pred = model_fn(window, days_ahead=days_ahead)
        except Exception:
            pred = float(window["close"].iloc[-1])
        actual = df["close"].iloc[i+days_ahead]
        prev = window["close"].iloc[-1]
        if (pred - prev) * (actual - prev) > 0:
            correct += 1
        total += 1
        errors.append(abs((pred - actual) / (actual + 1e-9)))
    acc = (correct / total) if total > 0 else 0.5
    mape = np.mean(errors) if errors else np.inf
    return {"acc": max(0.01, min(0.99, acc)), "mape": mape}

def dynamic_ensemble(df, days_ahead=3):
    models = {"hw": hw_predict, "ar1": ar1_predict, "fft": fft_predict, "trend": trend_extrapolate}
    stats = {k: model_recent_stats(df, m, days_ahead=days_ahead, lookback=90) for k,m in models.items()}
    scores = {}
    for k in stats:
        acc = stats[k]["acc"]
        mape = stats[k]["mape"]
        inv_mape = 1.0 / (mape + 1e-9) if np.isfinite(mape) and mape > 0 else 0.0
        scores[k] = acc**1.8 + 0.8 * inv_mape
    arr = np.array([max(1e-6, scores[k]) for k in models.keys()])
    weights = arr / arr.sum()
    preds = {k: float(models[k](df, days_ahead=days_ahead)) for k in models}
    pred = sum(preds[k] * weights[i] for i,k in enumerate(models.keys()))
    vol = df["ewma_vol"].iloc[-1] if "ewma_vol" in df else df["vol20"].iloc[-1]
    if vol < 0.02:
        w_adj = dict(zip(models.keys(), weights * (1 + np.array([0.05, -0.02, -0.01, 0.03]))))
        w_adj_arr = np.array([max(1e-6, w_adj[k]) for k in models.keys()])
        weights = w_adj_arr / w_adj_arr.sum()
        pred = sum(preds[k] * weights[i] for i,k in enumerate(models.keys()))
    sigma = vol * math.sqrt(max(1, days_ahead))
    lower = max(0.0, pred * (1 - sigma)); upper = pred * (1 + sigma)
    return float(pred), float(lower), float(upper)

# ---------- Sentiment & company info ----------
POS_WORDS = set(["good","beat","beats","growth","upgrade","strong","positive","outperform","gain","rise","record","profit","surge","beat estimates"])
NEG_WORDS = set(["loss","miss","misses","downgrade","weak","negative","underperform","drop","fall","concern","decline","cut","warn","missed"])

@lru_cache(maxsize=32)
def fetch_news_fmp(ticker_short: str, fmp_key: str, limit=40):
    if not fmp_key:
        return []
    try:
        url = f"https://financialmodelingprep.com/api/v3/stock_news"
        data = safe_get(url, params={"tickers": ticker_short, "limit": limit, "apikey": fmp_key})
        if isinstance(data, list):
            items = []
            for it in data[:limit]:
                items.append({"title": it.get("title",""), "text": it.get("text","")})
            return items
    except Exception:
        pass
    return []

@lru_cache(maxsize=32)
def fetch_news_yf(ticker: str, limit=40):
    if not HAVE_YFINANCE:
        return []
    try:
        t = yf.Ticker(ticker)
        news = getattr(t, "news", []) or []
        items = []
        for n in news[:limit]:
            items.append({"title": n.get("title",""), "text": n.get("summary","") or ""})
        return items
    except Exception:
        return []

def keyword_sentiment(news_items):
    if not news_items:
        return 0.0, "No recent news available."
    score = 0; count = 0
    for n in news_items[:40]:
        txt = (n.get("title","") + " " + n.get("text","")).lower()
        s = 0
        for w in POS_WORDS:
            if w in txt: s += 1
        for w in NEG_WORDS:
            if w in txt: s -= 1
        score += s; count += 1
    if count == 0:
        return 0.0, "No recent news available."
    val = float(max(-1.0, min(1.0, score / (4 * count))))
    if val > 0.15:
        desc = "News tone is generally positive."
    elif val < -0.15:
        desc = "News tone is generally negative — check headlines."
    else:
        desc = "News tone is mixed/neutral."
    return val, desc

def compute_vader_sentiment(news_items):
    try:
        if not VADER_AVAILABLE:
            return 0.0, "No recent news available."
        analyzer = SentimentIntensityAnalyzer()
        scores = []
        for it in (news_items or [])[:40]:
            txt = (it.get("title","") + ". " + it.get("text","")).strip()
            if not txt: continue
            vs = analyzer.polarity_scores(txt)
            scores.append(vs["compound"])
        if not scores:
            return 0.0, "No recent news available."
        mean = float(sum(scores)/len(scores))
        if mean > 0.2: desc = "News tone is generally positive."
        elif mean < -0.2: desc = "News tone is generally negative — check headlines."
        else: desc = "News tone is mixed/neutral."
        return mean, desc
    except Exception:
        return 0.0, "No recent news available."

def compute_sentiment(ticker_short: str, ticker_ns: str, fmp_key: str):
    if fmp_key:
        news = fetch_news_fmp(ticker_short, fmp_key, limit=40)
        if news:
            if VADER_AVAILABLE:
                return compute_vader_sentiment(news)
            return keyword_sentiment(news)
    news = fetch_news_yf(ticker_ns, limit=40)
    if news:
        if VADER_AVAILABLE:
            return compute_vader_sentiment(news)
        return keyword_sentiment(news)
    return 0.0, "No recent news available."

# ---------- Fundamentals fetch & scoring ----------
@lru_cache(maxsize=32)
def fetch_fundamentals(ticker: str, fmp_key: str = None):
    prof = {}
    if fmp_key:
        try:
            url = f"https://financialmodelingprep.com/api/v3/profile/{ticker.replace('.NS','')}"
            data = safe_get(url, params={"apikey": fmp_key})
            if isinstance(data, list) and len(data) > 0:
                return data[0]
        except Exception:
            pass
    if HAVE_YFINANCE:
        try:
            info = yf.Ticker(ticker).info or {}
            return info
        except Exception:
            pass
    return prof

def compute_fundamentals_score(profile):
    score = 40.0
    try:
        if isinstance(profile, dict) and profile:
            pe = None
            try:
                pe = float(profile.get("pe") or profile.get("trailingPE") or profile.get("forwardPE") or 0)
            except Exception:
                pe = None
            roe = None
            try:
                roe = float(profile.get("returnOnEquity") or profile.get("roe") or 0)
            except Exception:
                roe = None
            mcap = 0
            try:
                mcap = float(profile.get("mktCap") or profile.get("marketCap") or 0)
            except Exception:
                mcap = 0
            if pe and 0 < pe < 25:
                score += 22
            elif pe and pe > 25:
                score -= 5
            if roe and roe > 0.05:
                score += 20
            if mcap and mcap > 1e9:
                score += 8
    except Exception:
        pass
    return float(max(0, min(100, score)))

# ---------- Recommendation text ----------
def recommendation_text(pred, cur, lower, upper, implied_return, conf, sentiment_desc, momentum_score, fundamentals_score, trend):
    change = (pred / cur - 1) * 100
    lines = []
    lines.append(f"The model predicts ~{change:.2f}% change over your chosen horizon. Confidence: {conf:.2f}.")
    lines.append(f"Trend: {trend}. Momentum score: {momentum_score:.1f}/100. Fundamentals score: {fundamentals_score:.1f}/100.")
    lines.append(f"News: {sentiment_desc}")
    lines.append("")
    buy_price = cur * (1 - 0.03)
    strong_buy_price = cur * (1 - 0.08)
    take_profit = cur * (1 + max(0.06, change / 2 / 100))
    stop_loss = cur * (1 - 0.07)
    if change > 6 and conf > 0.55 and momentum_score > 55:
        lines.append("Recommendation: BUY (Reason: expected upside and supporting momentum).")
        lines.append(f"Suggested entry: consider buying near {buy_price:.2f} or in tranches. Strong entry if dips to {strong_buy_price:.2f}.")
        lines.append(f"Target / take-profit: {take_profit:.2f}. Stop-loss suggestion: {stop_loss:.2f}.")
    elif change > 2 and conf > 0.45:
        lines.append("Recommendation: CONSIDER BUY (small position).")
        lines.append(f"Suggested entry: small buy near {buy_price:.2f}. Target: {take_profit:.2f}. Stop-loss: {stop_loss:.2f}.")
    elif change < -4 and conf > 0.5:
        lines.append("Recommendation: SELL / AVOID NEW BUY (Reason: downside expected).")
        lines.append(f"If holding, consider trimming or set stop-loss near {stop_loss:.2f}.")
    else:
        lines.append("Recommendation: HOLD / WAIT (No clear edge).")
    lines.append("")
    lines.append(f"Practical predicted range: {lower:.2f} — {upper:.2f}. Implied return: {implied_return:.2f}%.")
    lines.append("Risk note: Use position sizing; VaR and CVaR below estimate potential shortfall levels for one day.")
    return "\n".join(lines)

# ---------- Main ----------
if submitted:
    ticker_input = ticker_raw.strip().upper()
    if "." not in ticker_input:
        ticker = append_nse_if_needed(ticker_input)
    else:
        ticker = ticker_input
    hist = None
    if HAVE_YFINANCE:
        hist = fetch_prices_yf(ticker, days=1500)
    if hist is None or len(hist) < 80:
        st.error("Not enough historical data. Ensure yfinance is installed and ticker is valid.")
    else:
        df = hist.rename(columns={"date":"date","close":"close","high":"high","low":"low","volume":"volume"}).sort_values("date").reset_index(drop=True)
        dfp = prepare_features(df)
        try:
            dfp["kf_close"] = kalman_smooth(dfp["close"])
        except Exception:
            dfp["kf_close"] = dfp["close"].rolling(3).mean()

        profile = fetch_fundamentals(ticker, fmp_key=fmp_key)
        fundamentals_score = compute_fundamentals_score(profile)

        try:
            mom = dfp["mom_5"].iloc[-1] if "mom_5" in dfp else 0.0
            rsi_val = dfp["rsi14"].iloc[-1] if "rsi14" in dfp else 50.0
            momentum_score = float(min(100, max(0, 50 + mom * 100 + (rsi_val - 50) * 0.4)))
        except Exception:
            momentum_score = 50.0

        daily_returns = df["close"].pct_change().dropna().values
        hist_var_95 = historical_var(daily_returns, alpha=0.05) if len(daily_returns)>0 else 0.0
        cvar_95 = cvar_expected_shortfall(daily_returns, alpha=0.05) if len(daily_returns)>0 else 0.0

        hurst = hurst_exponent(dfp["close"])
        halfl = half_life(dfp["close"])
        ssa_last, ssa_fore = ssa_trend_and_forecast(dfp["close"], window=90, components=3, days_ahead=MAP.get(interval,3))

        fetched_price = get_current_price_and_info(ticker_input, fmp_key=fmp_key)
        if fetched_price is None:
            cur = float(dfp["close"].iloc[-1])
        else:
            cur = float(fetched_price)

        days = MAP.get(interval, 3)
        pred, lower, upper = dynamic_ensemble(dfp, days_ahead=days)

        try:
            macd_hist = dfp["macd_hist"].iloc[-1] if "macd_hist" in dfp else 0.0
            sto_k = dfp["sto_k"].iloc[-1] if "sto_k" in dfp else 50.0
            ema_gap = (dfp["ema8"].iloc[-1] - dfp["ema21"].iloc[-1]) / (dfp["close"].iloc[-1] + 1e-9)
            entropy30 = dfp["entropy_30"].iloc[-1] if "entropy_30" in dfp else 0.0
            atr_rel = dfp["atr14"].iloc[-1] / (dfp["close"].iloc[-1] + 1e-9) if "atr14" in dfp else 0.01
            z = 0.6 * ema_gap + 0.25 * (macd_hist / (abs(dfp["macd_hist"].iloc[-60:].mean()) + 1e-9) if "macd_hist" in dfp else 0) + 0.15 * ((sto_k - 50) / 50)
            prob_up = 0.5 + 0.5 * np.tanh(z * 6)
            nudger = max(-0.06, min(0.06, (prob_up - 0.5) * 2.0 * 0.06))
            entropy_factor = -0.02 if entropy30 > 2.2 else 0.01
            pred = float(pred * (1.0 + nudger + entropy_factor))
        except Exception:
            pass

        implied_return = (pred / cur - 1) * 100.0
        trend = "Uptrend" if dfp["ema8"].iloc[-1] > dfp["ema21"].iloc[-1] else "Downtrend"
        conf = 0.5 + 0.28 * (1 - min(1, dfp["ewma_vol"].iloc[-1] * 10))
        conf = max(0.12, min(0.98, conf))

        ticker_short = ticker.replace(".NS","")
        sentiment_score, sentiment_desc = compute_sentiment(ticker_short, ticker, fmp_key)

        desc = ""
        try:
            if fmp_key:
                prof_fmp = safe_get(f"https://financialmodelingprep.com/api/v3/profile/{ticker_short}", params={"apikey": fmp_key})
                if isinstance(prof_fmp, list) and prof_fmp:
                    desc = prof_fmp[0].get("description","") or desc
        except Exception:
            desc = desc
        if not desc and HAVE_YFINANCE:
            try:
                info = yf.Ticker(ticker).info or {}
                desc = info.get("longBusinessSummary") or info.get("shortBusinessSummary") or desc
            except Exception:
                desc = desc
        if not desc:
            desc = "No company description available."

        final_score = 0.45 * (momentum_score / 100) + 0.35 * (fundamentals_score / 100) + 0.20 * conf
        final_score_pct = float(max(0, min(100, final_score * 100)))

        rec_text = recommendation_text(pred, cur, lower, upper, implied_return, conf, sentiment_desc, momentum_score, fundamentals_score, trend)

        # ---------- UI Display (only requested outputs) ----------
        st.markdown("<div style='max-width:1100px;margin:18px auto;padding:18px;border-radius:12px;box-shadow:0 12px 36px rgba(3,37,25,0.06);background:linear-gradient(180deg,#ffffff,#f7fff8);'>", unsafe_allow_html=True)
        left_col, right_col = st.columns([2,1])
        with left_col:
            st.markdown(f"<h3 style='color:#0B6E4F;margin-bottom:6px;'>Current Price</h3><p style='font-size:26px;margin-top:0;margin-bottom:6px;font-weight:700;'>{cur:.2f}</p>", unsafe_allow_html=True)
            st.markdown(f"<h3 style='color:#0B6E4F;margin-bottom:6px;'>Predicted Share Price ({interval})</h3><p style='font-size:24px;margin-top:0;margin-bottom:10px;font-weight:700;'>{pred:.2f}</p>", unsafe_allow_html=True)
            st.markdown(f"<h4 style='color:#0B6E4F;margin-bottom:6px;'>Practical Predicted Range</h4><p style='font-size:16px;margin-top:0;margin-bottom:6px;'>{lower:.2f} — {upper:.2f}</p>", unsafe_allow_html=True)
            st.markdown(f"<h4 style='color:#0B6E4F;margin-bottom:6px;'>Implied Return</h4><p style='font-size:16px;margin-top:0;margin-bottom:6px;'>{implied_return:.2f}%</p>", unsafe_allow_html=True)
        with right_col:
            chart_df = pd.DataFrame({
                "metric":["Momentum","Fundamentals","Confidence","Final Score"],
                "value":[momentum_score, fundamentals_score, conf*100, final_score_pct]
            })
            try:
                import altair as alt
                c = alt.Chart(chart_df).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
                    x=alt.X("value:Q", title=""),
                    y=alt.Y("metric:N", sort="-x", title=""),
                    color=alt.condition(alt.datum.value > 60, alt.value("#0b6e4f"), alt.value("#7f8c8d"))
                ).properties(height=160, width=320)
                st.altair_chart(c, use_container_width=True)
            except Exception:
                for i,row in chart_df.iterrows():
                    st.markdown(f"**{row['metric']}**")
                    st.progress(min(100, max(0, int(row['value']))))
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div style='max-width:1100px;margin:12px auto;padding:18px;border-radius:12px;background:linear-gradient(180deg,#ffffff,#fbfff9);box-shadow:0 10px 28px rgba(6,45,32,0.04);'>", unsafe_allow_html=True)
        cols = st.columns([2,1])
        with cols[0]:
            st.markdown(f"<h3 style='color:#0B6E4F;margin-top:2px;margin-bottom:6px;'>Company Description</h3>", unsafe_allow_html=True)
            st.markdown(f"<p style='font-size:14px;color:#333;margin-top:0;margin-bottom:8px;line-height:1.4'>{desc}</p>", unsafe_allow_html=True)
            st.markdown(f"<h3 style='color:#0B6E4F;margin-top:6px;margin-bottom:6px;'>Sentiment</h3>", unsafe_allow_html=True)
            st.markdown(f"<p style='font-size:14px;color:#333;margin-top:0;margin-bottom:8px;line-height:1.4'>{sentiment_desc}</p>", unsafe_allow_html=True)
        with cols[1]:
            st.markdown(f"<h3 style='color:#0B6E4F;margin-top:2px;margin-bottom:6px;'>Confidence Level</h3>", unsafe_allow_html=True)
            st.markdown(f"<p style='font-size:18px;color:#333;margin-top:0;margin-bottom:8px; font-weight:700'>{conf:.2f}</p>", unsafe_allow_html=True)
            st.markdown(f"<h3 style='color:#0B6E4F;margin-top:6px;margin-bottom:6px;'>Risk Controls (1-day)</h3>", unsafe_allow_html=True)
            st.markdown(f"<p style='font-size:14px;color:#333;margin-top:0;margin-bottom:8px;'>Historical VaR(95%): {hist_var_95:.4f} | CVaR(95%): {cvar_95:.4f}</p>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='color:#0B6E4F;margin-top:8px;margin-bottom:6px;'>Recommendation (Beginner-friendly)</h3>", unsafe_allow_html=True)
        st.markdown(f"<pre style='white-space:pre-wrap;font-size:14px;color:#222;background:transparent;border:none;padding:0;margin:0'>{rec_text}</pre>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
