# ajinkya_predictor_v2.1_full.py
# Ajinkya's Predictor V2.1 — Complete ready-to-paste app (single file)
# Run: pip install -r requirements.txt && streamlit run ajinkya_predictor_v2.1_full.py

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

st.set_page_config(page_title="Ajinkya's Predictor V2.1", layout="wide")

# Header UI (only display name and subtitle)
st.markdown("""
<div style="background:linear-gradient(90deg,#082f22,#0b6e4f); padding:14px; border-radius:12px; color:white; display:flex; align-items:center; gap:12px;">
  <div style="width:48px;height:48px;background:linear-gradient(135deg,#66cdaa,#0b6e4f);border-radius:8px;display:flex;align-items:center;justify-content:center;font-weight:800;color:#042f1b;">
    AJ
  </div>
  <div>
    <div style="font-size:24px;font-weight:800;">Ajinkya's Predictor V2.1</div>
    <div style="font-size:11px;color:#dff6ea;margin-top:3px;">Improved accuracy • Robust fetchers • Advanced math</div>
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

# ---------- Utilities & fetchers ----------
def append_nse_if_needed(ticker: str):
    t = ticker.strip().upper()
    if "." in t: return t
    return t + ".NS"

def safe_get(url, params=None, timeout=8):
    try:
        r = requests.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None

@lru_cache(maxsize=64)
def fetch_prices_yf(ticker: str, days=1500):
    if not HAVE_YFINANCE: return None
    try:
        tk = yf.Ticker(ticker)
        end = datetime.utcnow()
        start = end - timedelta(days=int(days*1.1))
        df = tk.history(start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"), interval="1d", actions=False)
        if df is None or df.empty: return None
        df = df.reset_index().rename(columns={"Date":"date","Close":"close","High":"high","Low":"low","Volume":"volume"})
        df["date"] = pd.to_datetime(df["date"]); df = df.sort_values("date").reset_index(drop=True)
        return df[["date","close","high","low","volume"]]
    except Exception:
        return None

# Price fetchers (FMP -> Yahoo)
@lru_cache(maxsize=128)
def try_fmp_quote(ticker_short: str, fmp_key: str):
    if not fmp_key: return None
    try:
        for tk in (ticker_short, ticker_short + ".NS"):
            data = safe_get(f"https://financialmodelingprep.com/api/v3/quote/{tk}", params={"apikey": fmp_key})
            if isinstance(data, list) and len(data) > 0:
                item = data[0]
                price = item.get("price") or item.get("close") or item.get("previousClose")
                if price is not None:
                    try: return float(price)
                    except: pass
    except Exception:
        pass
    return None

@lru_cache(maxsize=128)
def try_yf_current(ticker_ns: str):
    if not HAVE_YFINANCE: return None
    try:
        tk = yf.Ticker(ticker_ns)
        fast = getattr(tk, "fast_info", None)
        if isinstance(fast, dict):
            for k in ("last_price","lastPrice","last_trade_price","last"):
                if k in fast and fast[k] is not None:
                    try: return float(fast[k])
                    except: pass
        # intraday 1m
        try:
            hist = tk.history(period="2d", interval="1m", actions=False)
            if hist is not None and len(hist) > 0:
                last = hist["Close"].iloc[-1]
                if not pd.isna(last): return float(last)
        except Exception:
            pass
        # fallback to daily close
        try:
            histd = tk.history(period="7d", interval="1d", actions=False)
            if histd is not None and len(histd) > 0:
                last = histd["Close"].iloc[-1]
                if not pd.isna(last): return float(last)
        except Exception:
            pass
    except Exception:
        pass
    return None

@lru_cache(maxsize=128)
def get_current_price(ticker_input: str, fmp_key: str = None):
    if not ticker_input: return None
    t_input = ticker_input.strip().upper()
    t_short = t_input.replace(".NS", "")
    t_ns = t_short + ".NS" if ".NS" not in t_input else t_input
    # try FMP first
    if fmp_key:
        p = try_fmp_quote(t_short, fmp_key)
        if p is not None: return p
        p = try_fmp_quote(t_short + ".NS", fmp_key)
        if p is not None: return p
    # then Yahoo
    p = try_yf_current(t_ns)
    if p is not None: return p
    return None

# ---------- Math & indicators ----------
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

def macd(series):
    return macd_components(series)

def atr(df, window=14):
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    return tr.rolling(window).mean().fillna(method="bfill")

def bollinger(series, w=20, nstd=2):
    m = series.rolling(w).mean()
    sd = series.rolling(w).std()
    return m, m + nstd * sd, m - nstd * sd

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

def kalman_smooth(series):
    n = len(series); xhat = np.zeros(n); P = np.zeros(n)
    Q = 1e-5; R = np.var(series) * 0.01 + 1e-6
    xhat[0] = series.iloc[0]; P[0] = 1.0
    for k in range(1, n):
        xhatminus = xhat[k-1]; Pminus = P[k-1] + Q
        K = Pminus / (Pminus + R)
        xhat[k] = xhatminus + K * (series.iloc[k] - xhatminus); P[k] = (1 - K) * Pminus
    return pd.Series(xhat, index=series.index)

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

def rolling_ols_slope(series, window=30):
    return series.rolling(window).apply(lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if x.size >= 3 else 0, raw=False).fillna(0)

# SSA utilities (simplified)
import numpy.linalg as npl
def ssa_trend_and_forecast(series, window=60, components=3, days_ahead=3):
    try:
        x = np.array(series.dropna())[-(window+0):]
        n = len(x)
        if n < 2 * components:
            return float(series.iloc[-1]), float(series.iloc[-1])
        L = max(2, n // 2)
        K = n - L + 1
        X = np.column_stack([x[i:i+K] for i in range(L)])
        U, s, Vt = npl.svd(X, full_matrices=False)
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

def seasonal_strength(series, max_lag=60):
    try:
        s = series.dropna().values
        n = len(s)
        if n < 30:
            return 0.0
        s = (s - np.mean(s)) / (np.std(s) + 1e-9)
        acfs = [1.0] + [np.corrcoef(s[:-k], s[k:])[0,1] for k in range(1, min(max_lag, n//2))]
        acfs = np.nan_to_num(np.array(acfs))
        peak = np.max(np.abs(acfs[1:]))
        return float(peak)
    except Exception:
        return 0.0

def volatility_regime_detector(df, kurt_window=30):
    try:
        r = df["close"].pct_change().dropna()
        if len(r) < kurt_window:
            return "UNKNOWN"
        rk = r.rolling(kurt_window).kurt().iloc[-1]
        vratio = r.rolling(kurt_window).var().iloc[-1] / (r.rolling(kurt_window*2).var().iloc[-1] + 1e-9)
        if rk > 5 or vratio > 1.5:
            return "TURBULENT"
        if rk < 2 and vratio < 1.1:
            return "CALM"
        return "MIXED"
    except Exception:
        return "UNKNOWN"

def earnings_revision_proxy(profile_fmp, profile_yf):
    try:
        delta = 0.0
        if profile_fmp and isinstance(profile_fmp, dict):
            if profile_fmp.get("eps") and profile_fmp.get("epsTrailing"):
                delta = float(profile_fmp.get("eps")) - float(profile_fmp.get("epsTrailing"))
        if profile_yf and isinstance(profile_yf, dict) and delta == 0.0:
            fpe = profile_yf.get("forwardPE"); tpe = profile_yf.get("trailingPE")
            if fpe and tpe:
                delta = float(tpe) - float(fpe)
        if delta > 0:
            return "POSITIVE"
        if delta < 0:
            return "NEGATIVE"
        return "NEUTRAL"
    except Exception:
        return "NEUTRAL"

def piotroski_f_score(profile_fmp, profile_yf=None):
    score = 0
    try:
        net_income = None
        roa = None
        if profile_fmp and isinstance(profile_fmp, dict):
            net_income = profile_fmp.get("netIncome", profile_fmp.get("netIncomeBasic"))
            roa = profile_fmp.get("returnOnAssets") or profile_fmp.get("returnOnEquity")
        if profile_yf and isinstance(profile_yf, dict):
            if net_income is None:
                net_income = profile_yf.get("netIncomeToCommon")
            if roa is None:
                roa = profile_yf.get("returnOnAssets") or profile_yf.get("returnOnEquity")
        if net_income and float(net_income) > 0:
            score += 1
        if roa and float(roa) > 0:
            score += 1
        try:
            de = None
            if profile_yf:
                de = profile_yf.get("debtToEquity") or None
            if de and float(de) < 1.0:
                score += 1
        except Exception:
            pass
        try:
            if profile_yf:
                cr = profile_yf.get("currentRatio")
                if cr and float(cr) > 1.0:
                    score += 1
        except Exception:
            pass
        try:
            if profile_yf:
                om = profile_yf.get("operatingMargins") or profile_yf.get("grossMargins")
                if om and float(om) > 0.05:
                    score += 1
        except Exception:
            pass
    except Exception:
        pass
    return int(max(0, min(9, score * 1)))

def half_life(series):
    try:
        series = series.dropna()
        if len(series) < 30:
            return np.nan
        x = np.log(series)
        if len(x) < 10:
            return np.nan
        phi = np.corrcoef(x.values[:-1], x.values[1:])[0,1]
        if phi >= 1:
            return np.inf
        halflife = -np.log(2) / np.log(abs(phi)) if phi != 0 else np.inf
        return float(max(0.0, halflife))
    except Exception:
        return np.nan

def detect_trend(df):
    try:
        ema8 = df["ema8"].iloc[-1]; ema21 = df["ema21"].iloc[-1]; sma50 = df["sma50"].iloc[-1]; price = df["close"].iloc[-1]
        if price > ema8 > ema21 and price > sma50: return "Strong Uptrend"
        if ema8 > ema21 and price >= sma50: return "Uptrend"
        if ema8 < ema21 and price < sma50: return "Downtrend"
        return "Sideways / Uncertain"
    except Exception:
        return "Unknown"

# ---------- Feature engineering helpers ----------
def sample_skewness(series):
    x = np.array(series.dropna()); n = len(x)
    if n < 3: return 0.0
    m = x.mean(); s = x.std(ddof=1)
    return float((n/((n-1)*(n-2))) * np.sum(((x-m)/s)**3))

def sample_kurtosis(series):
    x = np.array(series.dropna()); n = len(x)
    if n < 4: return 0.0
    m = x.mean(); s = x.std(ddof=1)
    g2 = (n*(n+1))/((n-1)*(n-2)*(n-3)) * np.sum(((x-m)/s)**4) - 3*(n-1)**2/((n-2)*(n-3))
    return float(g2)

def shannon_entropy(series, bins=20):
    try:
        hist, _ = np.histogram(series.dropna(), bins=bins, density=True)
        probs = hist/(hist.sum()+1e-9)
        probs = probs[probs>0]
        ent = -np.sum(probs * np.log(probs + 1e-12))
        return float(ent)
    except Exception:
        return 0.0

# Prepare features (full)
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
    df["atr14"] = atr(df, window=14)
    df["bb_mid"], df["bb_upper"], df["bb_lower"] = bollinger(df["close"], w=20, nstd=2)
    df["entropy_30"] = df["close"].rolling(30).apply(lambda x: shannon_entropy(pd.Series(x)), raw=False).fillna(0)
    df["skew30"] = df["return1"].rolling(30).apply(lambda x: sample_skewness(pd.Series(x)), raw=False).fillna(0)
    df["kurt30"] = df["return1"].rolling(30).apply(lambda x: sample_kurtosis(pd.Series(x)), raw=False).fillna(0)
    for lag in [1,2,3,5,8,13,21]:
        df[f"lag_{lag}"] = df["close"].shift(lag)
    df["mom_5"] = df["close"].pct_change(5); df["mom_21"] = df["close"].pct_change(21)
    df = df.dropna().reset_index(drop=True)
    return df

# ---------- Forecast building blocks ----------
def hw_predict(df, days_ahead=3):
    try:
        if not HAVE_STATSMODELS: raise Exception("no statsmodels")
        model = ExponentialSmoothing(df["close"], trend="add", seasonal=None, damped_trend=True)
        fit = model.fit(optimized=True)
        return float(fit.forecast(days_ahead).iloc[-1])
    except Exception:
        return float(df["close"].iloc[-1])

def ar1_predict(df, days_ahead=3):
    try:
        r = df["logret"].dropna()
        if len(r) < 10: return float(df["close"].iloc[-1])
        phi = np.corrcoef(r[:-1], r[1:])[0,1] if len(r)>1 else 0
        mu = r.mean(); last = r.iloc[-1]
        fore = mu + phi*(last-mu)
        return float(df["close"].iloc[-1] * math.exp(fore * days_ahead))
    except Exception:
        return float(df["close"].iloc[-1])

def fft_predict(df, days_ahead=3):
    try:
        return float(fft_dominant_forecast(df["close"], days_ahead=days_ahead, top_k=6))
    except Exception:
        return float(df["close"].iloc[-1])

def trend_predict(df, days_ahead=3):
    try:
        slope = df["close"].rolling(30).apply(lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x)>=3 else 0).iloc[-1]
        return float(df["close"].iloc[-1] + slope * days_ahead)
    except Exception:
        return float(df["close"].iloc[-1])

# Model performance on recent history for dynamic weighting
def model_recent_stats(df, model_fn, days_ahead=3, lookback=90):
    n = len(df)
    if n < 40: return {"acc":0.5, "mape":np.inf}
    lookback = min(lookback, n - days_ahead - 5)
    start = n - lookback - days_ahead
    correct = 0; total = 0; errors = []
    for i in range(start, n - days_ahead):
        window = df.iloc[:i+1]
        try: pred = model_fn(window, days_ahead=days_ahead)
        except: pred = float(window["close"].iloc[-1])
        actual = df["close"].iloc[i+days_ahead]; prev = window["close"].iloc[-1]
        if (pred - prev) * (actual - prev) > 0: correct += 1
        total += 1; errors.append(abs((pred - actual) / (actual + 1e-9)))
    acc = (correct / total) if total>0 else 0.5; mape = np.mean(errors) if errors else np.inf
    return {"acc": max(0.01, min(0.99, acc)), "mape": mape}

def dynamic_ensemble(df, days_ahead=3):
    models = {"hw":hw_predict, "ar1":ar1_predict, "fft":fft_predict, "trend":trend_predict}
    stats = {k: model_recent_stats(df, m, days_ahead=days_ahead, lookback=90) for k,m in models.items()}
    scores = {}
    for k in stats:
        acc = stats[k]["acc"]; mape = stats[k]["mape"]
        inv_mape = 1.0 / (mape + 1e-9) if np.isfinite(mape) and mape>0 else 0.0
        scores[k] = acc**1.8 + 0.8 * inv_mape
    arr = np.array([max(1e-6, scores[k]) for k in models.keys()])
    weights = arr / arr.sum()
    preds = {k: float(models[k](df, days_ahead=days_ahead)) for k in models}
    pred = sum(preds[k] * weights[i] for i,k in enumerate(models.keys()))
    vol = df["ewma_vol"].iloc[-1] if "ewma_vol" in df else df["vol20"].iloc[-1]
    sigma = vol * math.sqrt(max(1, days_ahead))
    lower = max(0.0, pred * (1 - sigma)); upper = pred * (1 + sigma)
    return float(pred), float(lower), float(upper)

# ---------- Risk & scoring ----------
def historical_var(returns, alpha=0.05):
    if len(returns) < 10: return 0.0
    q = np.percentile(returns, alpha * 100)
    return float(max(0.0, -q))

def parametric_var(returns, alpha=0.05):
    if len(returns) < 10: return 0.0
    mu = np.mean(returns); sigma = np.std(returns)
    z = np.percentile(np.random.normal(size=200000), alpha * 100)
    var = -(mu + sigma * z)
    return float(max(0.0, var))

def cvar_expected_shortfall(returns, alpha=0.05):
    if len(returns) < 10: return 0.0
    threshold = np.quantile(returns, alpha)
    tail = returns[returns <= threshold]
    if len(tail) == 0: return 0.0
    return float(max(0.0, -np.mean(tail)))

# ---------- Sentiment & company description ----------
POS = set(["good","beat","beats","growth","upgrade","strong","positive","outperform","gain","rise","record","profit","surge"])
NEG = set(["loss","miss","misses","downgrade","weak","negative","underperform","drop","fall","concern","decline","cut","warn"])

@lru_cache(maxsize=32)
def fetch_news_yf(ticker: str, limit=40):
    if not HAVE_YFINANCE: return []
    try:
        t = yf.Ticker(ticker); news = getattr(t, "news", []) or []
        items = []
        for n in news[:limit]:
            items.append({"title": n.get("title",""), "text": n.get("summary","") or ""})
        return items
    except Exception:
        return []

def keyword_sentiment(news_items):
    if not news_items: return 0.0, "No recent news available."
    score = 0; count = 0
    for n in news_items[:40]:
        txt = (n.get("title","") + " " + n.get("text","")).lower(); s = 0
        for w in POS: 
            if w in txt: s += 1
        for w in NEG:
            if w in txt: s -= 1
        score += s; count += 1
    if count == 0: return 0.0, "No recent news available."
    val = float(max(-1.0, min(1.0, score / (4 * count))))
    if val > 0.15: desc = "News tone is generally positive."
    elif val < -0.15: desc = "News tone is generally negative — check headlines."
    else: desc = "News tone is mixed/neutral."
    return val, desc

def compute_vader(news_items):
    try:
        if not VADER_AVAILABLE: return keyword_sentiment(news_items)
        analyzer = SentimentIntensityAnalyzer(); scores = []
        for it in news_items[:40]:
            txt = (it.get("title","") + " " + it.get("text","")).strip()
            if not txt: continue
            vs = analyzer.polarity_scores(txt); scores.append(vs["compound"])
        if not scores: return 0.0, "No recent news available."
        mean = float(sum(scores) / len(scores))
        if mean > 0.2: desc = "News tone is generally positive."
        elif mean < -0.2: desc = "News tone is generally negative — check headlines."
        else: desc = "News tone is mixed/neutral."
        return mean, desc
    except Exception:
        return keyword_sentiment(news_items)

def compute_sentiment(ticker_short: str, ticker_ns: str, fmp_key: str):
    # Try FMP news if key provided
    if fmp_key:
        try:
            data = safe_get("https://financialmodelingprep.com/api/v3/stock_news", params={"tickers": ticker_short, "limit": 40, "apikey": fmp_key})
            if isinstance(data, list) and data:
                news = [{"title": d.get("title",""), "text": d.get("text","")} for d in data]
                return compute_vader(news) if VADER_AVAILABLE else keyword_sentiment(news)
        except Exception:
            pass
    # fallback yfinance news
    news = fetch_news_yf(ticker_ns, limit=40)
    if news:
        return compute_vader(news) if VADER_AVAILABLE else keyword_sentiment(news)
    return 0.0, "No recent news available."

@lru_cache(maxsize=32)
def fetch_company_description(fmp_key: str, ticker: str):
    desc = ""
    if fmp_key:
        try:
            url = f"https://financialmodelingprep.com/api/v3/profile/{ticker}"
            data = safe_get(url, params={"apikey": fmp_key})
            if data and isinstance(data, list) and len(data)>0:
                d = data[0]; desc = d.get("description") or d.get("companyName") or ""
                if desc: return desc
        except Exception:
            pass
    if HAVE_YFINANCE:
        try:
            info = yf.Ticker(ticker).info
            desc = info.get("longBusinessSummary") or info.get("shortBusinessSummary") or ""
            if desc: return desc
        except Exception:
            pass
    return desc or "No company description available."

# ---------- Screeners & scoring ----------
def compute_momentum_score(df):
    ret = df["close"].pct_change(5).iloc[-1]
    ema_slope = (df["ema8"].iloc[-1] - df["ema8"].iloc[-12]) if len(df) >= 12 else 0
    r = df["rsi14"].iloc[-1] if "rsi14" in df else 50
    score = 0.4*(np.tanh(ret*10)) + 0.35*(np.tanh(ema_slope/(df["close"].iloc[-1]+1e-9))) + 0.25*((r-50)/50)
    return float(max(0, min(100, (score+1)/2*100)))

@lru_cache(maxsize=32)
def fetch_fundamentals(ticker: str, fmp_key: str = None):
    prof = {}
    if fmp_key:
        try:
            data = safe_get(f"https://financialmodelingprep.com/api/v3/profile/{ticker.replace('.NS','')}", params={"apikey": fmp_key})
            if isinstance(data, list) and data: return data[0]
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
            try: pe = float(profile.get("pe") or profile.get("trailingPE") or profile.get("forwardPE") or 0)
            except: pe = None
            roe = None
            try: roe = float(profile.get("returnOnEquity") or profile.get("roe") or 0)
            except: roe = None
            mcap = 0
            try: mcap = float(profile.get("mktCap") or profile.get("marketCap") or 0)
            except: mcap = 0
            if pe and 0 < pe < 25: score += 22
            elif pe and pe > 25: score -= 5
            if roe and roe > 0.05: score += 20
            if mcap and mcap > 1e9: score += 8
    except Exception:
        pass
    return float(max(0, min(100, score)))

def run_screeners(profile_fmp, profile_yf, momentum_score, fundamentals_score, vol, daily_returns):
    # Value screener
    value_pass = False
    pe = None
    roe = None
    try:
        if profile_fmp and isinstance(profile_fmp, dict):
            pe = profile_fmp.get("pe") or profile_fmp.get("trailingPE") or profile_fmp.get("forwardPE")
            roe = profile_fmp.get("returnOnEquity") or profile_fmp.get("roe")
        if profile_yf and isinstance(profile_yf, dict):
            if pe is None: pe = profile_yf.get("trailingPE") or profile_yf.get("forwardPE")
            if roe is None: roe = profile_yf.get("returnOnEquity") or profile_yf.get("roe")
    except Exception:
        pe = pe; roe = roe
    try:
        if pe and float(pe) > 0 and float(pe) < 25 and roe and float(roe) > 0.08:
            value_pass = True
    except Exception:
        value_pass = False

    # Momentum screener
    momentum_pass = momentum_score >= 55

    # Volatility screener: low VaR desirable for "safe" buy
    hist_var = historical_var(daily_returns, alpha=0.05) if len(daily_returns)>0 else 0.0
    vol_pass = hist_var < 0.03  # tuned threshold (example)

    # Quality screener
    quality_pass = fundamentals_score >= 60

    # Dividend screener
    div_pass = False
    try:
        dy = None
        if profile_yf and isinstance(profile_yf, dict):
            dy = profile_yf.get("dividendYield")
        if profile_fmp and isinstance(profile_fmp, dict):
            if dy is None: dy = profile_fmp.get("lastDividend") or profile_fmp.get("dividendYield")
        if dy and float(dy) > 0.015:
            div_pass = True
    except Exception:
        div_pass = False

    components = [value_pass, momentum_pass, vol_pass, quality_pass, div_pass]
    screener_score = sum([1 if c else 0 for c in components]) / max(1, len(components))
    screener_influence = (screener_score - 0.5) * 0.12  # -0.06 .. +0.06
    badges = {
        "Value": ("PASS" if value_pass else "NO"),
        "Momentum": ("PASS" if momentum_pass else "WEAK"),
        "Volatility (VaR)": ("LOW" if vol_pass else "HIGH"),
        "Quality": ("PASS" if quality_pass else "NO"),
        "Dividend": ("YES" if div_pass else "NO"),
        "Score": round(screener_score, 2)
    }
    return badges, screener_influence

# ---------- Recommendation & text ----------
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
        lines.append("If you want to enter, prefer small positions and tight stop-losses; wait for clearer trend confirmation.")
    lines.append("")
    lines.append(f"Practical predicted range: {lower:.2f} — {upper:.2f}. Implied return: {implied_return:.2f}%.")
    lines.append("Risk note: Use position sizing; VaR and CVaR below estimate potential shortfall levels for one day.")
    return "\n".join(lines)

# ---------- Main run ----------
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
        # smoothing
        try:
            dfp["kf_close"] = kalman_smooth(dfp["close"])
        except Exception:
            dfp["kf_close"] = dfp["close"].rolling(3).mean()

        # fundamentals (yfinance fallback)
        profile_yf = None
        if HAVE_YFINANCE:
            try:
                profile_yf = yf.Ticker(ticker).info or {}
            except Exception:
                profile_yf = None
        profile_fmp = None
        if fmp_key:
            try:
                prof_fmp = safe_get(f"https://financialmodelingprep.com/api/v3/profile/{ticker.replace('.NS','')}", params={"apikey": fmp_key})
                if isinstance(prof_fmp, list) and prof_fmp:
                    profile_fmp = prof_fmp[0]
            except Exception:
                profile_fmp = None

        fundamentals_score = compute_fundamentals_score(profile_yf or profile_fmp or {})
        momentum_score = compute_momentum_score(dfp) if len(dfp)>0 else 50.0

        # risk & returns
        daily_returns = df["close"].pct_change().dropna()
        hist_var_95 = historical_var(daily_returns.values, alpha=0.05) if len(daily_returns)>0 else 0.0
        param_var_95 = parametric_var(daily_returns.values, alpha=0.05)
        cvar_95 = cvar_expected_shortfall(daily_returns.values, alpha=0.05)

        # screeners
        badges, screener_influence = run_screeners(profile_fmp, profile_yf, momentum_score, fundamentals_score, dfp["ewma_vol"].iloc[-1], daily_returns.values)

        # advanced math & screeners
        market_close = None
        try:
            if HAVE_YFINANCE:
                m_t = yf.Ticker("^NSEI")
                m_hist = m_t.history(period="3y", interval="1d")
                if m_hist is not None and not m_hist.empty:
                    market_close = m_hist["Close"].reset_index(drop=True)
        except Exception:
            market_close = None

        beta = rolling_beta = None
        try:
            if market_close is not None:
                rets_x = df["close"].pct_change().dropna()
                rets_m = market_close.pct_change().dropna()
                series = pd.concat([rets_x, rets_m], axis=1).dropna()
                if len(series) >= 60:
                    betas = series.iloc[:,0].rolling(60).cov(series.iloc[:,1]) / (series.iloc[:,1].rolling(60).var() + 1e-12)
                    beta = float(betas.iloc[-1])
        except Exception:
            beta = np.nan

        piotroski = piotroski_f_score(profile_fmp or {}, profile_yf or {})
        hl = half_life(dfp["close"])
        ssa_last, ssa_fore = ssa_trend_and_forecast(dfp["close"], window=90, components=3, days_ahead=MAP.get(interval,3))
        seasonal_strength_val = seasonal_strength(dfp["close"], max_lag=66)
        vol_regime = volatility_regime_detector(dfp)
        earnings_rev = earnings_revision_proxy(profile_fmp or {}, profile_yf or {})

        beta_badge = "HIGH" if (not np.isnan(beta) and abs(beta) > 1.2) else ("LOW" if not np.isnan(beta) else "UNKNOWN")
        piot_badge = "STRONG" if piotroski >= 6 else ("AVERAGE" if piotroski >= 3 else "POOR")
        hl_badge = "FAST" if (not np.isnan(hl) and hl < 10) else ("SLOW" if not np.isnan(hl) else "UNKNOWN")
        ssa_badge = "GOOD" if abs(ssa_last - dfp["close"].iloc[-1]) / (dfp["close"].iloc[-1] + 1e-9) < 0.03 else "POOR"
        seasonal_badge = "STRONG" if seasonal_strength_val > 0.25 else "WEAK"
        earn_badge = earnings_rev

        badges.update({
            "Beta": beta_badge,
            "Piotroski": piot_badge,
            "HalfLife": hl_badge,
            "SSA": ssa_badge,
            "Seasonality": seasonal_badge,
            "EarningsRev": earn_badge,
            "VolRegime": vol_regime
        })

        # current price (fallback to history close)
        fetched_price = get_current_price(ticker_input, fmp_key)
        if fetched_price is None:
            cur = float(dfp["close"].iloc[-1])
        else:
            cur = float(fetched_price)

        # predict
        days = MAP.get(interval, 3)
        pred, lower, upper = dynamic_ensemble(dfp, days_ahead=days)

        # small directional nudge
        try:
            ema_gap = (dfp["ema8"].iloc[-1] - dfp["ema21"].iloc[-1]) / (dfp["close"].iloc[-1] + 1e-9)
            macd_hist = dfp["macd_hist"].iloc[-1] if "macd_hist" in dfp else 0.0
            nudger = max(-0.06, min(0.06, 0.5 * np.tanh(ema_gap*8) + 0.3 * np.tanh(macd_hist/ (abs(dfp['macd_hist'].iloc[-60:].mean())+1e-9) if "macd_hist" in dfp else 0)))
            pred = float(pred * (1.0 + nudger))
        except Exception:
            pass

        implied_return = (pred / cur - 1) * 100.0
        trend = detect_trend(dfp)
        conf = 0.5 + 0.28 * (1 - min(1, dfp["ewma_vol"].iloc[-1] * 10))
        conf = max(0.12, min(0.98, conf))

        # sentiment & desc
        news_items = fetch_news_yf(ticker) if HAVE_YFINANCE else []
        sentiment_score, sentiment_desc = compute_sentiment(ticker.replace(".NS",""), ticker, fmp_key)
        desc = fetch_company_description(fmp_key, ticker)

        final_score = 0.45 * (momentum_score / 100) + 0.35 * (fundamentals_score / 100) + 0.20 * conf
        final_score_pct = float(max(0, min(100, final_score * 100)))

        rec_text = recommendation_text(pred, cur, lower, upper, implied_return, conf, sentiment_desc, momentum_score, fundamentals_score, trend)

        # ---------- UI display ----------
        st.markdown("<div style='max-width:1100px;margin:18px auto;padding:18px;border-radius:12px;background:linear-gradient(180deg,#ffffff,#f7fff8);'>", unsafe_allow_html=True)
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
                    st.write(f"**{row['metric']}**")
                    st.progress(min(100, max(0, int(row['value']))))
            st.markdown("<div style='margin-top:10px; display:flex; flex-wrap:wrap; gap:8px;'>", unsafe_allow_html=True)
            def badge_html(name, label):
                color = "#2ecc71" if label in ("PASS","LOW","YES","GOOD","STRONG") else ("#f1c40f" if label in ("WEAK","NO","AVERAGE","MIXED","SLOW") else "#e74c3c")
                return f"<div style='background:{color};padding:6px 10px;border-radius:8px;color:white;font-weight:600;font-size:13px;'>{name}: {label}</div>"
            st.markdown(badge_html("Value", badges["Value"]), unsafe_allow_html=True)
            st.markdown(badge_html("Momentum", badges["Momentum"]), unsafe_allow_html=True)
            st.markdown(badge_html("Volatility (VaR)", badges["Volatility (VaR)"]), unsafe_allow_html=True)
            st.markdown(badge_html("Quality", badges["Quality"]), unsafe_allow_html=True)
            st.markdown(badge_html("Dividend", badges["Dividend"]), unsafe_allow_html=True)
            # extra badges
            st.markdown(badge_html("Beta", badges["Beta"]), unsafe_allow_html=True)
            st.markdown(badge_html("Piotroski", badges["Piotroski"]), unsafe_allow_html=True)
            st.markdown(badge_html("HalfLife", badges["HalfLife"]), unsafe_allow_html=True)
            st.markdown(badge_html("SSA", badges["SSA"]), unsafe_allow_html=True)
            st.markdown(badge_html("Seasonality", badges["Seasonality"]), unsafe_allow_html=True)
            st.markdown(badge_html("EarningsRev", badges["EarningsRev"]), unsafe_allow_html=True)
            st.markdown(badge_html("VolRegime", badges["VolRegime"]), unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div style='max-width:1100px;margin:12px auto;padding:18px;border-radius:12px;background:linear-gradient(180deg,#ffffff,#fbfff9);'>", unsafe_allow_html=True)
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
            st.markdown(f"<p style='font-size:14px;color:#333;margin-top:0;margin-bottom:8px;'>Historical VaR(95%): {hist_var_95:.4f} | Parametric VaR(95%): {param_var_95:.4f} | CVaR(95%): {cvar_95:.4f}</p>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='color:#0B6E4F;margin-top:8px;margin-bottom:6px;'>Recommendation (Beginner-friendly)</h3>", unsafe_allow_html=True)
        st.markdown(f"<pre style='white-space:pre-wrap;font-size:14px;color:#222;background:transparent;border:none;padding:0;margin:0'>{rec_text}</pre>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
# End of file
