
# Ajinkya's Predictor V2.0 - More Math Tools
# Adds ATR, Bollinger Bands, Keltner Channels, realized volatility, Shannon entropy, skewness, kurtosis,
# integrates them into feature prep and ensemble nudging. Keeps UI unchanged and hides internals.
import streamlit as st
import pandas as pd
import numpy as np
import math, requests, warnings
from datetime import datetime, timedelta
from functools import lru_cache

warnings.filterwarnings("ignore")

try:
    import yfinance as yf; HAVE_YFINANCE = True
except Exception:
    HAVE_YFINANCE = False

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing; HAVE_STATSMODELS = True
except Exception:
    HAVE_STATSMODELS = False

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer; VADER_AVAILABLE = True
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
    <div style="font-size:12px; color:#dff6ea; margin-top:4px;">More Math — ATR, Bollinger, Keltner, Entropy, Skew/Kurtosis</div>
  </div>
</div>
""", unsafe_allow_html=True)

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

def safe_get(url, params=None, timeout=6):
    try:
        r = requests.get(url, params=params or {}, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None

@lru_cache(maxsize=64)
def fetch_prices_yf(ticker: str, days=1500):
    if not HAVE_YFINANCE: return None
    try:
        t = yf.Ticker(ticker)
        end = datetime.now(); start = end - timedelta(days=int(days*1.1))
        df = t.history(start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"), interval="1d", actions=False)
        if df is None or df.empty: return None
        df = df.reset_index().rename(columns={"Date":"date","Close":"close","High":"high","Low":"low","Volume":"volume"})
        df["date"] = pd.to_datetime(df["date"]); df = df.sort_values("date").reset_index(drop=True)
        return df[["date","close","high","low","volume"]]
    except Exception:
        return None

@lru_cache(maxsize=128)
def get_current_price(ticker_input: str, fmp_key: str = None):
    if not ticker_input: return None
    t_input = ticker_input.strip().upper(); t_short = t_input.replace(".NS",""); t_ns = t_short + ".NS" if ".NS" not in t_input else t_input
    # try FMP
    if fmp_key:
        try:
            url = f"https://financialmodelingprep.com/api/v3/quote/{t_short}"
            jd = safe_get(url, params={"apikey": fmp_key})
            if isinstance(jd, list) and jd:
                it = jd[0]; price = it.get("price") or it.get("close") or it.get("previousClose")
                if price is not None: return float(price)
        except Exception: pass
    # yfinance
    if HAVE_YFINANCE:
        try:
            tk = yf.Ticker(t_ns)
            fast = getattr(tk, "fast_info", None)
            if isinstance(fast, dict):
                for fld in ("last_price","lastPrice","last_trade_price","last"):
                    if fld in fast and fast[fld] is not None:
                        try: return float(fast[fld])
                        except: pass
            try:
                hist = tk.history(period="2d", interval="1m", actions=False)
                if hist is not None and len(hist)>0:
                    last = hist["Close"].iloc[-1]
                    if not pd.isna(last): return float(last)
            except Exception: pass
            try:
                histd = tk.history(period="7d", interval="1d", actions=False)
                if histd is not None and len(histd)>0:
                    last = histd["Close"].iloc[-1]; 
                    if not pd.isna(last): return float(last)
            except Exception: pass
        except Exception: pass
    return None

# --- Additional math tools ---
def atr(df, window=14):
    """Average True Range"""
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
    rv = returns.rolling(window).apply(lambda x: np.sqrt(np.sum(x**2) * (252/len(x))) if len(x)>0 else 0).fillna(0)
    return rv

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

# reuse indicators from prior builds: ema, rsi, macd, ssa, kalman etc.
def ema(series, span): return series.ewm(span=span, adjust=False).mean()
def rsi(series, period=14):
    delta = series.diff(); up = delta.clip(lower=0).fillna(0); down = -1*delta.clip(upper=0).fillna(0)
    ma_up = up.ewm(alpha=1.0/period, adjust=False).mean(); ma_down = down.ewm(alpha=1.0/period, adjust=False).mean()
    rs = ma_up / ma_down.replace(0, np.nan); rsi = 100 - (100/(1+rs)); return rsi.fillna(50)

def ssa_trend_and_forecast(series, window=60, components=3, days_ahead=3):
    try:
        x = np.array(series.dropna())[-window:]; n = len(x)
        if n < 2*components: return float(series.iloc[-1]), float(series.iloc[-1])
        L = max(2, n//2); K = n - L + 1
        X = np.column_stack([x[i:i+K] for i in range(L)])
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        Xr = sum(s[i] * np.outer(U[:,i], Vt[i,:]) for i in range(min(components, len(s))))
        recon = np.zeros(n); counts = np.zeros(n)
        for i in range(L):
            for j in range(K):
                recon[i+j] += Xr[i,j]; counts[i+j] += 1
        recon = recon / (counts + 1e-9)
        t = np.arange(len(recon)); p = np.polyfit(t, recon, 1)
        forecast = np.polyval(p, len(recon)+days_ahead-1)
        return float(recon[-1]), float(forecast)
    except Exception:
        return float(series.iloc[-1]), float(series.iloc[-1])

# feature prep with new math tools
def prepare_features(df):
    df = df.copy()
    df["close"] = df["close"].astype(float)
    df["return1"] = df["close"].pct_change().fillna(0)
    df["logret"] = np.log(df["close"]/df["close"].shift(1)).replace([np.inf,-np.inf],0).fillna(0)
    df["ema8"] = ema(df["close"], 8); df["ema21"] = ema(df["close"], 21); df["sma50"] = df["close"].rolling(50).mean()
    df["vol20"] = df["return1"].rolling(20).std().fillna(0); df["ewma_vol"] = df["return1"].ewm(span=20, adjust=False).std().fillna(0)
    df["rsi14"] = rsi(df["close"]); df["mom_5"] = df["close"].pct_change(5)
    df["atr14"] = atr(df, window=14); df["bb_mid"], df["bb_upper"], df["bb_lower"] = bollinger_bands(df["close"], window=20, n_std=2)
    df["kelt_mid"], df["kelt_upper"], df["kelt_lower"] = keltner_channel(df, ema_window=20, atr_window=10, mult=1.5)
    df["rv21"] = realized_volatility(df["return1"], window=21)
    df["entropy_30"] = df["close"].rolling(30).apply(lambda x: shannon_entropy(pd.Series(x)), raw=False).fillna(0)
    df["skew30"] = df["return1"].rolling(30).apply(lambda x: sample_skewness(pd.Series(x)), raw=False).fillna(0)
    df["kurt30"] = df["return1"].rolling(30).apply(lambda x: sample_kurtosis(pd.Series(x)), raw=False).fillna(0)
    df["macd"], df["macd_sig"], df["macd_hist"] = df["close"].ewm(span=12, adjust=False).mean() - df["close"].ewm(span=26, adjust=False).mean(), None, None
    # compute MACD components safely
    macd_line = df["close"].ewm(span=12, adjust=False).mean() - df["close"].ewm(span=26, adjust=False).mean()
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    df["macd"] = macd_line; df["macd_sig"] = macd_signal; df["macd_hist"] = df["macd"] - df["macd_sig"]
    # drop nans and lags
    for lag in [1,2,3,5,8,13,21]:
        df[f"lag_{lag}"] = df["close"].shift(lag)
    df = df.dropna().reset_index(drop=True)
    return df

# simple forecasts used in ensemble
def hw_predict(df, days_ahead=3):
    try:
        if not HAVE_STATSMODELS: raise Exception("no statsmodels")
        model = ExponentialSmoothing(df["close"], trend="add", seasonal=None, damped_trend=True)
        fitted = model.fit(optimized=True)
        return float(fitted.forecast(days_ahead).iloc[-1])
    except Exception:
        return float(df["close"].iloc[-1])

def ar1_predict(df, days_ahead=3):
    try:
        r = df["logret"].dropna()
        if len(r)<10: return float(df["close"].iloc[-1])
        phi = np.corrcoef(r[:-1], r[1:])[0,1]
        mu = r.mean(); last = r.iloc[-1]
        fore = mu + phi*(last-mu)
        return float(df["close"].iloc[-1] * math.exp(fore*days_ahead))
    except Exception:
        return float(df["close"].iloc[-1])

def fft_predict(df, days_ahead=3):
    try:
        x = df["close"].values; n = len(x)
        if n<12: return float(df["close"].iloc[-1])
        t = np.arange(n); p = np.polyfit(t, x, 1); trend = np.polyval(p, t); resid = x-trend
        fft = np.fft.rfft(resid); freqs = np.fft.rfftfreq(n); amps = np.abs(fft)
        idx = np.argsort(amps)[-6:]; future_t = np.arange(n, n+days_ahead); recon = np.zeros(days_ahead)
        for i in idx:
            a = fft[i]; freq = freqs[i]; phase = np.angle(a); amplitude = np.abs(a)/n*2
            recon += amplitude * np.cos(2*np.pi*freq*future_t + phase)
        last_trend = np.polyval(p, n-1); slope = p[0]; trend_fore = last_trend + slope*np.arange(1, days_ahead+1)
        return float((trend_fore + recon)[-1])
    except Exception:
        return float(df["close"].iloc[-1])

def trend_extrapolate(df, days_ahead=3):
    slope = df["close"].rolling(30).apply(lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x)>=3 else 0).iloc[-1]
    return float(df["close"].iloc[-1] + slope*days_ahead)

# ensemble combining directional accuracy and inverse MAPE (like before)
def model_recent_stats(df, model_fn, days_ahead=3, lookback=90):
    n = len(df); 
    if n < 40: return {"acc":0.5, "mape":np.inf}
    if n < lookback + days_ahead + 5: lookback = max(30, n//3)
    start = n - lookback - days_ahead; correct=0; total=0; errors=[]
    for i in range(start, n - days_ahead):
        window = df.iloc[:i+1]
        try: pred = model_fn(window, days_ahead=days_ahead)
        except: pred = float(window["close"].iloc[-1])
        actual = df["close"].iloc[i+days_ahead]; prev = window["close"].iloc[-1]
        if (pred-prev)*(actual-prev)>0: correct+=1
        total+=1; errors.append(abs((pred-actual)/(actual+1e-9)))
    acc = (correct/total) if total>0 else 0.5; mape = np.mean(errors) if errors else np.inf
    return {"acc": max(0.01, min(0.99, acc)), "mape": mape}

def dynamic_ensemble(df, days_ahead=3):
    models = {"hw":hw_predict, "ar1":ar1_predict, "fft":fft_predict, "trend":trend_extrapolate}
    stats = {k: model_recent_stats(df, m, days_ahead=days_ahead, lookback=90) for k,m in models.items()}
    scores = {}
    for k in stats:
        acc = stats[k]["acc"]; mape = stats[k]["mape"]
        inv_mape = 1.0/(mape+1e-9) if np.isfinite(mape) and mape>0 else 0.0
        scores[k] = acc**1.8 + 0.8*inv_mape
    arr = np.array([max(1e-6, scores[k]) for k in models.keys()])
    weights = arr / arr.sum()
    preds = {k: float(models[k](df, days_ahead=days_ahead)) for k in models}
    pred = sum(preds[k]*weights[i] for i,k in enumerate(models.keys()))
    vol = df["ewma_vol"].iloc[-1] if "ewma_vol" in df else df["vol20"].iloc[-1]
    if vol < 0.02:
        w_adj = dict(zip(models.keys(), weights*(1+np.array([0.05,-0.02,-0.01,0.03]))))
        w_adj_arr = np.array([max(1e-6, w_adj[k]) for k in models.keys()])
        weights = w_adj_arr / w_adj_arr.sum()
        pred = sum(preds[k]*weights[i] for i,k in enumerate(models.keys()))
    sigma = vol * math.sqrt(max(1, days_ahead))
    lower = max(0.0, pred*(1-sigma)); upper = pred*(1+sigma)
    return float(pred), float(lower), float(upper)

# sentiment & fundamentals (reuse simple fallbacks)
@lru_cache(maxsize=32)
def fetch_news_yf(ticker: str, limit: int = 40):
    if not HAVE_YFINANCE: return []
    try:
        t = yf.Ticker(ticker); news = getattr(t, "news", []) or []
        items = []
        for n in news[:limit]: items.append({"title": n.get("title",""), "text": n.get("summary","") or ""})
        return items
    except Exception: return []

def keyword_sentiment(news_items):
    POS = set(["good","beat","beats","growth","upgrade","strong","positive","outperform","gain","rise","record","profit","surge"])
    NEG = set(["loss","miss","misses","downgrade","weak","negative","underperform","drop","fall","concern","decline","cut","warn"])
    if not news_items: return 0.0, "No recent news available."
    score=0; count=0
    for n in news_items[:40]:
        txt = ((n.get("title","")+" "+n.get("text","")).lower()); s=0
        for w in POS:
            if w in txt: s+=1
        for w in NEG:
            if w in txt: s-=1
        score+=s; count+=1
    if count==0: return 0.0, "No recent news available."
    val = float(max(-1.0, min(1.0, score/(4*count))))
    desc = "News tone is mixed/neutral."
    if val>0.15: desc = "News tone is generally positive."
    elif val < -0.15: desc = "News tone is generally negative — check headlines."
    return val, desc

def compute_sentiment(ticker, fmp_key):
    news = fetch_news_yf(ticker)
    if news:
        if VADER_AVAILABLE:
            analyzer = SentimentIntensityAnalyzer(); scores=[]
            for it in news[:40]:
                txt = (it.get("title","")+" "+it.get("text","")).strip()
                if not txt: continue
                vs = analyzer.polarity_scores(txt); scores.append(vs["compound"])
            if scores:
                mean = float(sum(scores)/len(scores))
                if mean>0.2: desc="News tone is generally positive."
                elif mean<-0.2: desc="News tone is generally negative — check headlines."
                else: desc="News tone is mixed/neutral."
                return mean, desc
        return keyword_sentiment(news)
    return 0.0, "No recent news available."

@lru_cache(maxsize=32)
def fetch_fundamentals(ticker: str, fmp_key: str = None):
    prof = {}
    if HAVE_YFINANCE:
        try: return yf.Ticker(ticker).info or {}
        except Exception: pass
    return prof

def compute_fundamentals_score(profile):
    score = 40.0
    try:
        if isinstance(profile, dict):
            pe = profile.get("pe") or profile.get("trailingPE") or profile.get("forwardPE")
            roe = profile.get("returnOnEquity") or profile.get("roe")
            mcap = profile.get("mktCap") or profile.get("marketCap") or 0
            if pe and isinstance(pe,(int,float)) and 0<pe<30: score+=20
            if roe and isinstance(roe,(int,float)) and roe>0.05: score+=20
            if mcap and mcap>1e9: score+=10
    except Exception: pass
    return float(max(0, min(100, score)))

def recommendation_text(pred, cur, lower, upper, implied_return, conf, sentiment_desc, momentum_score, fundamentals_score, trend):
    change = (pred/cur-1)*100; lines=[]
    lines.append(f"The model predicts ~{change:.2f}% change over your chosen horizon. Confidence: {conf:.2f}.")
    lines.append(f"Trend: {trend}. Momentum score: {momentum_score:.1f}/100. Fundamentals score: {fundamentals_score:.1f}/100.")
    lines.append(f"News: {sentiment_desc}"); lines.append("")
    buy_price = cur*(1-0.03); strong_buy_price = cur*(1-0.08)
    take_profit = cur*(1+max(0.06, change/2/100)); stop_loss = cur*(1-0.07)
    if change>6 and conf>0.55 and momentum_score>55:
        lines.append("Recommendation: BUY (Reason: expected upside and supporting momentum).")
        lines.append(f"Suggested entry: consider buying near {buy_price:.2f} or in tranches. Strong entry if dips to {strong_buy_price:.2f}.")
        lines.append(f"Target / take-profit: {take_profit:.2f}. Stop-loss suggestion: {stop_loss:.2f}.")
    elif change>2 and conf>0.45:
        lines.append("Recommendation: CONSIDER BUY (small position)."); lines.append(f"Suggested entry: small buy near {buy_price:.2f}. Target: {take_profit:.2f}. Stop-loss: {stop_loss:.2f}.")
    elif change < -4 and conf > 0.5:
        lines.append("Recommendation: SELL / AVOID NEW BUY (Reason: downside expected)."); lines.append(f"If holding, consider trimming or set stop-loss near {stop_loss:.2f}.")
    else:
        lines.append("Recommendation: HOLD / WAIT (No clear edge).")
    lines.append(""); lines.append(f"Practical predicted range: {lower:.2f} — {upper:.2f}. Implied return: {implied_return:.2f}%.")
    lines.append("Risk note: Use position sizing; VaR and CVaR below estimate potential shortfall levels for one day.")
    return "\n".join(lines)

if submitted:
    ticker_input = ticker_raw.strip().upper(); ticker = append_nse_if_needed(ticker_input) if "." not in ticker_input else ticker_input
    hist = fetch_prices_yf(ticker, days=1500) if HAVE_YFINANCE else None
    if hist is None or len(hist) < 60:
        st.error("Not enough historical data. Ensure yfinance is installed and ticker is valid.")
    else:
        df = hist.sort_values("date").reset_index(drop=True)
        dfp = prepare_features(df)
        dfp["kf_close"] = dfp["close"].rolling(5).mean()  # simple smooth
        profile = fetch_fundamentals(ticker, fmp_key=fmp_key); fundamentals_score = compute_fundamentals_score(profile)
        momentum_score = float(min(100, max(0, 50 + dfp["mom_5"].iloc[-1]*100)))
        daily_returns = df["close"].pct_change().dropna().values
        hist_var_95 = historical_var(daily_returns, alpha=0.05) if len(daily_returns)>0 else 0.0
        cvar_95 = cvar_expected_shortfall(daily_returns, alpha=0.05) if len(daily_returns)>0 else 0.0
        # extra indicators for UI/weighting
        atr14 = dfp["atr14"].iloc[-1] if "atr14" in dfp else 0.0
        bb_mid = dfp["bb_mid"].iloc[-1] if "bb_mid" in dfp else dfp["close"].iloc[-1]
        kelt_mid = dfp["kelt_mid"].iloc[-1] if "kelt_mid" in dfp else dfp["close"].iloc[-1]
        entropy30 = dfp["entropy_30"].iloc[-1] if "entropy_30" in dfp else 0.0
        skew30 = dfp["skew30"].iloc[-1] if "skew30" in dfp else 0.0
        kurt30 = dfp["kurt30"].iloc[-1] if "kurt30" in dfp else 0.0
        # price fetch
        fetched_price = get_current_price(ticker_input, fmp_key=fmp_key); cur = float(fetched_price) if fetched_price is not None else float(dfp["close"].iloc[-1])
        days = MAP.get(interval, 3); pred, lower, upper = dynamic_ensemble(dfp, days_ahead=days)
        # nudge using additional math indicators (ATR relative, entropy low -> higher confidence)
        atr_nudge = max(-0.04, min(0.04, (dfp["atr14"].iloc[-1] / (dfp["close"].iloc[-1]+1e-9) - 0.01) * 2))
        ent_nudge = 0.02 if entropy30 < 2.0 else -0.02
        pred = float(pred * (1 + atr_nudge + ent_nudge))
        implied_return = (pred / cur - 1) * 100.0
        trend = "Uptrend" if dfp["ema8"].iloc[-1] > dfp["ema21"].iloc[-1] else "Downtrend"
        conf = max(0.12, min(0.98, 0.5 + 0.2*(1 - min(1, dfp["ewma_vol"].iloc[-1]*10)) + (0.05 if entropy30<1.8 else 0)))
        sentiment_score, sentiment_desc = compute_sentiment(ticker, fmp_key)
        desc = profile.get("description","") if isinstance(profile, dict) else ""
        # UI display (unchanged fields)
        st.markdown("<div style='max-width:1100px;margin:18px auto;padding:18px;border-radius:12px;background:#fff;'>", unsafe_allow_html=True)
        left_col, right_col = st.columns([2,1])
        with left_col:
            st.markdown(f"<h3 style='color:#0B6E4F;margin-bottom:6px;'>Current Price</h3><p style='font-size:26px;margin-top:0;margin-bottom:6px;font-weight:700;'>{cur:.2f}</p>", unsafe_allow_html=True)
            st.markdown(f"<h3 style='color:#0B6E4F;margin-bottom:6px;'>Predicted Share Price ({interval})</h3><p style='font-size:24px;margin-top:0;margin-bottom:10px;font-weight:700;'>{pred:.2f}</p>", unsafe_allow_html=True)
            st.markdown(f"<h4 style='color:#0B6E4F;margin-bottom:6px;'>Practical Predicted Range</h4><p style='font-size:16px;margin-top:0;margin-bottom:6px;'>{lower:.2f} — {upper:.2f}</p>", unsafe_allow_html=True)
            st.markdown(f"<h4 style='color:#0B6E4F;margin-bottom:6px;'>Implied Return</h4><p style='font-size:16px;margin-top:0;margin-bottom:6px;'>{implied_return:.2f}%</p>", unsafe_allow_html=True)
        with right_col:
            st.markdown(f"<h3 style='color:#0B6E4F;margin-bottom:6px;'>Momentum</h3><p style='font-size:16px;margin-top:0;margin-bottom:6px;'>{momentum_score:.1f}/100</p>", unsafe_allow_html=True)
            st.markdown(f"<h3 style='color:#0B6E4F;margin-bottom:6px;'>Fundamentals</h3><p style='font-size:16px;margin-top:0;margin-bottom:6px;'>{fundamentals_score:.1f}/100</p>", unsafe_allow_html=True)
            st.markdown(f"<h3 style='color:#0B6E4F;margin-bottom:6px;'>Confidence</h3><p style='font-size:16px;margin-top:0;margin-bottom:6px;font-weight:700'>{conf:.2f}</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("<div style='max-width:1100px;margin:12px auto;padding:18px;border-radius:12px;background:#fff;'>", unsafe_allow_html=True)
        cols = st.columns([2,1])
        with cols[0]:
            st.markdown("<h3 style='color:#0B6E4F;margin-top:2px;margin-bottom:6px;'>Company Description</h3>", unsafe_allow_html=True)
            st.markdown(f"<p style='font-size:14px;color:#333;margin-top:0;margin-bottom:8px;line-height:1.4'>{desc}</p>", unsafe_allow_html=True)
            st.markdown("<h3 style='color:#0B6E4F;margin-top:6px;margin-bottom:6px;'>Sentiment</h3>", unsafe_allow_html=True)
            st.markdown(f"<p style='font-size:14px;color:#333;margin-top:0;margin-bottom:8px;line-height:1.4'>{sentiment_desc}</p>", unsafe_allow_html=True)
        with cols[1]:
            st.markdown("<h3 style='color:#0B6E4F;margin-top:2px;margin-bottom:6px;'>Risk Controls (1-day)</h3>", unsafe_allow_html=True)
            st.markdown(f"<p style='font-size:14px;color:#333;margin-top:0;margin-bottom:8px;'>Historical VaR(95%): {hist_var_95:.4f} | CVaR(95%): {cvar_95:.4f}</p>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='color:#0B6E4F;margin-top:8px;margin-bottom:6px;'>Recommendation (Beginner-friendly)</h3>", unsafe_allow_html=True)
        rec = recommendation_text(pred, cur, lower, upper, implied_return, conf, sentiment_desc, momentum_score, fundamentals_score, trend)
        st.markdown(f"<pre style='white-space:pre-wrap;font-size:14px;color:#222;background:transparent;border:none;padding:0;margin:0'>{rec}</pre>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

