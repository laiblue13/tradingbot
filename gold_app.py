from datetime import datetime
import json, os
from sklearn.ensemble import HistGradientBoostingClassifier

import numpy as np, pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf

st.set_page_config(
    page_title="AI Gold Reverse XAUUSD ($100)", layout="wide", page_icon="🥇"
)

st.markdown(
    """
<style>
    * { opacity: 1 !important; transition: none !important; }
    [data-testid="stStatusWidget"], .stSpinner { display: none !important; }
</style>
""",
    unsafe_allow_html=True,
)

st.title("🥇 Bot AI Giao Dịch Vàng XAUUSD - Chế Độ Đánh Ngược (Vốn $100)")
ACC_FILE = "gold_demo_account.json"
ALL_TFS = ["1m", "5m", "15m", "30m", "1h"]
GOLD_TICKER = "GC=F"


def load_acc():
  today = datetime.now().strftime("%Y-%m-%d")
  if os.path.exists(ACC_FILE):
    with open(ACC_FILE, "r", encoding="utf-8") as f:
      acc = json.load(f)
      if acc.get("date") != today:
        acc["date"], acc["day_start"] = today, acc["bal"]
      return acc
  return {
      "init": 100.0,
      "bal": 100.0,
      "date": today,
      "day_start": 100.0,
      "pos": None,
      "hist": [],
  }


def save_acc(acc):
  with open(ACC_FILE, "w", encoding="utf-8") as f:
    json.dump(acc, f, indent=2, ensure_ascii=False)


with st.sidebar:
  st.header("🤖 Bảng Điều Khiển XAUUSD")
  auto_trade = st.toggle("🟢 BẬT TỰ ĐỘNG QUÉT & VÀO LỆNH", True)
  reverse_mode = st.toggle("🔄 ĐÁNH NGƯỢC AI (AI Buy -> Bot Sell)", True)
  scan_tfs = st.multiselect(
      "Các khung cho phép Bot săn lệnh:", ALL_TFS, default=ALL_TFS
  )
  chart_tf = st.selectbox("Khung biểu đồ hiển thị:", ALL_TFS, index=1)
  mtf_filter = st.checkbox("Lọc sóng lớn (1H EMA 50)", False)
  be_mode = st.checkbox("Dời SL hòa vốn khi lãi 1R", True)
  risk_pct = st.slider("Rủi ro mỗi lệnh (%):", 1.0, 5.0, 2.0, 0.5)
  max_loss_pct = st.slider("Khóa bot nếu lỗ ngày (%):", 3.0, 15.0, 6.0, 1.0)
  conf_th = st.slider("Độ tự tin AI tối thiểu (%):", 40, 75, 48, 2)
  if st.button("🔄 Reset về $100 Demo", use_container_width=True):
    if os.path.exists(ACC_FILE):
      os.remove(ACC_FILE)
    st.rerun()


def clean_yf_df(raw_df):
  if isinstance(raw_df.columns, pd.MultiIndex):
    raw_df.columns = raw_df.columns.get_level_values(0)
  df = raw_df[["Open", "High", "Low", "Close", "Volume"]].copy()
  df.columns = ["open", "high", "low", "close", "volume"]
  if df.index.tz is not None:
    df.index = df.index.tz_convert("Asia/Ho_Chi_Minh").tz_localize(None)
  else:
    df.index = df.index + pd.Timedelta(hours=7)
  return df[~df.index.duplicated(keep="last")].dropna()


@st.cache_data(ttl=60)
def analyze_xau_tf(tf):
  raw_1h = yf.download(
      GOLD_TICKER, period="15d", interval="1h", progress=False
  )
  df_1h = clean_yf_df(raw_1h)
  macro = (
      "UP"
      if float(df_1h["close"].iloc[-1])
      > float(df_1h["close"].ewm(span=50, adjust=False).mean().iloc[-1])
      else "DOWN"
  )

  period_map = {
      "1m": "7d",
      "5m": "30d",
      "15m": "60d",
      "30m": "60d",
      "1h": "730d",
  }
  tg_map = {
      "1m": 0.00025,
      "5m": 0.0006,
      "15m": 0.0010,
      "30m": 0.0015,
      "1h": 0.0022,
  }

  raw = yf.download(
      GOLD_TICKER, period=period_map[tf], interval=tf, progress=False
  )
  df = clean_yf_df(raw)
  df["hr"] = df.index.hour

  df["HA_c"] = (df["open"] + df["high"] + df["low"] + df["close"]) / 4
  ha_o = [(float(next(iter(df["open"]))) + float(next(iter(df["close"])))) / 2]
  hac_list = df["HA_c"].tolist()
  for i in range(1, len(df)):
    ha_o.append((ha_o[i - 1] + hac_list[i - 1]) / 2)
  df["HA_o"] = ha_o
  df["HA_h"] = df[["high", "HA_o", "HA_c"]].max(axis=1)
  df["HA_l"] = df[["low", "HA_o", "HA_c"]].min(axis=1)

  df["col"] = np.where(df["HA_c"] > df["HA_o"], 1, -1)
  df["body"] = (df["HA_c"] - df["HA_o"]) / df["HA_o"]
  df["uw"] = (df["HA_h"] - df[["HA_o", "HA_c"]].max(axis=1)) / df["HA_o"]
  df["lw"] = (df[["HA_o", "HA_c"]].min(axis=1) - df["HA_l"]) / df["HA_o"]
  df["streak"] = df["col"] * (
      df["col"].groupby((df["col"] != df["col"].shift()).cumsum()).cumcount()
      + 1
  )
  df["vol_r"] = df["volume"] / (df["volume"].rolling(20).mean() + 1e-9)

  d = df["HA_c"].diff()
  df["rsi"] = 100 - (
      100
      / (
          1
          + d.where(d > 0, 0).ewm(alpha=1 / 14, adjust=False).mean()
          / (-d.where(d < 0, 0)).ewm(alpha=1 / 14, adjust=False).mean()
      )
  )
  df["ema20"] = df["HA_c"].ewm(span=20, adjust=False).mean()
  df["dist_ema"] = (df["HA_c"] - df["ema20"]) / df["ema20"]
  df["ATR"] = (
      pd.concat(
          [
              df["high"] - df["low"],
              (df["high"] - df["close"].shift(1)).abs(),
              (df["low"] - df["close"].shift(1)).abs(),
          ],
          axis=1,
      )
      .max(axis=1)
      .rolling(14)
      .mean()
  )

  po, pc, co, cc = (
      df["open"].shift(1),
      df["close"].shift(1),
      df["open"],
      df["close"],
  )
  df["bull_e"] = (
      (pc < po) & (cc > co) & (co <= pc) & (cc >= po)
  ).astype(int)
  df["bear_e"] = (
      (pc > po) & (cc < co) & (co >= pc) & (cc <= po)
  ).astype(int)

  ret = df["close"].shift(-4) / df["close"] - 1
  df["target"] = np.where(
      ret > tg_map[tf], 1, np.where(ret < -tg_map[tf], 2, 0)
  )
  clean = df.dropna().copy()

  feats = [
      "hr",
      "col",
      "body",
      "uw",
      "lw",
      "streak",
      "vol_r",
      "rsi",
      "dist_ema",
      "bull_e",
      "bear_e",
  ]
  sp = int(len(clean) * 0.8)
      model = HistGradientBoostingClassifier(
        class_weight="balanced",
        max_iter=150,
        learning_rate=0.05,
        max_depth=5,
        random_state=42,
  )

  model.fit(clean[feats].iloc[:sp], clean["target"].iloc[:sp])

  rec = df.iloc[-80:].copy()
  probs = model.predict_proba(rec[feats])
  return rec, probs[-1], macro
def get_real_spot_xauusd():
  try:
    url = "https://swissquote.com"
    res = requests.get(url, timeout=3).json()
    top_quote = next(iter(res))
    price_info = next(iter(top_quote["spreadProfilePrices"]))
    bid = float(price_info["bid"])
    ask = float(price_info["ask"])
    return round((bid + ask) / 2, 2)
  except Exception:
    raw = yf.download(GOLD_TICKER, period="1d", interval="1m", progress=False)
    return round(float(clean_yf_df(raw)["close"].iloc[-1]), 2)


@st.fragment(run_every=2)
def live_xau_panel():
  acc = load_acc()
  live_price = get_real_spot_xauusd()
  now_s = datetime.now().strftime("%H:%M:%S")

  scan_results = []
  best_signal = None
  macro_trend = "UP"

  for t in scan_tfs:
    rec, probs, macro = analyze_xau_tf(t)
    macro_trend = macro
    raw_sig, conf = int(np.argmax(probs)), float(np.max(probs) * 100)
    atr_val = float(rec["ATR"].iloc[-1])

    # Đảo ngược tín hiệu AI nếu bật chế độ Đánh Ngược
    sig = raw_sig
    if reverse_mode:
      if raw_sig == 1:
        sig = 2
      elif raw_sig == 2:
        sig = 1

    ok_dir = (
        (not mtf_filter)
        or (sig == 1 and macro == "UP")
        or (sig == 2 and macro == "DOWN")
    )
    status = "🟡 Chờ"
    if sig == 1 and conf >= conf_th:
      status = (
          ("🟢 BUY (Đảo)" if reverse_mode else "🟢 BUY")
          if ok_dir
          else "⚠️ Ngược 1H"
      )
    elif sig == 2 and conf >= conf_th:
      status = (
          ("🔴 SELL (Đảo)" if reverse_mode else "🔴 SELL")
          if ok_dir
          else "⚠️ Ngược 1H"
      )

    scan_results.append({
        "Khung": t,
        "Tín hiệu": status,
        "Độ tự tin": f"{conf:.1f}%",
        "ATR ($)": round(atr_val, 2),
    })

    if (sig == 1 or sig == 2) and conf >= conf_th and ok_dir:
      if best_signal is None or conf > best_signal["conf"]:
        best_signal = {"tf": t, "sig": sig, "conf": conf, "atr": atr_val}

  pos, fpnl = acc["pos"], 0.0
  if pos:
    diff = (
        (live_price - pos["entry"])
        if pos["type"] == "BUY"
        else (pos["entry"] - live_price)
    )
    fpnl = (diff * pos["oz"]) - (pos["entry"] * pos["oz"] * 0.0002)
    if be_mode and not pos.get("be") and fpnl >= pos["risk"]:
      pos["sl"], pos["be"] = pos["entry"], True
      save_acc(acc)

    reason = None
    if pos["type"] == "BUY":
      if live_price >= pos["tp"]:
        reason = "Chốt lời 2R 🎯"
      elif live_price <= pos["sl"]:
        reason = "Hòa vốn 🛡️" if pos.get("be") else "Cắt lỗ 1R 🛑"
    else:
      if live_price <= pos["tp"]:
        reason = "Chốt lời 2R 🎯"
      elif live_price >= pos["sl"]:
        reason = "Hòa vốn 🛡️" if pos.get("be") else "Cắt lỗ 1R 🛑"

    if reason:
      acc["bal"] = max(0.0, acc["bal"] + fpnl)
      acc["hist"].insert(
          0,
          {
              "Khung": pos.get("tf", "5m"),
              "Lệnh": pos["type"],
              "Vào": f"${pos['entry']:,.2f}",
              "Đóng": f"${live_price:,.2f}",
              "Lãi/Lỗ ($)": round(fpnl, 2),
              "Lý do": reason,
              "Giờ": now_s,
          },
      )
      acc["pos"], pos, fpnl = None, None, 0.0
      save_acc(acc)

  day_pnl = (acc["bal"] + fpnl) - acc["day_start"]
  locked = day_pnl <= -(acc["day_start"] * (max_loss_pct / 100.0))

  if (
      not pos
      and auto_trade
      and not locked
      and acc["bal"] >= 10.0
      and best_signal
  ):
    risk_u = acc["bal"] * (risk_pct / 100.0)
    sl_d = max(best_signal["atr"] * 1.2, 0.5)
    l_type = "BUY" if best_signal["sig"] == 1 else "SELL"
    tp_p = (
        round(live_price + sl_d * 2, 2)
        if l_type == "BUY"
        else round(live_price - sl_d * 2, 2)
    )
    sl_p = (
        round(live_price - sl_d, 2)
        if l_type == "BUY"
        else round(live_price + sl_d, 2)
    )

    acc["pos"] = pos = {
        "tf": best_signal["tf"],
        "type": l_type,
        "entry": live_price,
        "tp": tp_p,
        "sl": sl_p,
        "oz": risk_u / sl_d,
        "risk": risk_u,
        "be": False,
        "time": now_s,
    }
    save_acc(acc)

  eq = acc["bal"] + fpnl
  c1, c2, c3, c4, c5 = st.columns(5)
  c1.metric(f"🔴 SPOT XAUUSD ({now_s})", f"${live_price:,.2f}", "Live Forex")
  c2.metric("💰 Vốn khả dụng", f"${acc['bal']:,.2f}")
  c3.metric("🏦 Tổng tài sản", f"${eq:,.2f}", f"{eq - acc['init']:+,.2f} USD")
  c4.metric(
      "🧭 Sóng lớn 1H",
      "🟢 TĂNG" if macro_trend == "UP" else "🔴 GIẢM",
      f"Ngày: {day_pnl:+,.2f}$",
  )

  if locked:
    c5.error("🛑 KHÓA LỖ NGÀY")
  elif best_signal:
    c5.success(
        f"🔥 BẮT KÈO {best_signal['tf']}: {'BUY' if best_signal['sig']==1 else 'SELL'} ({best_signal['conf']:.1f}%)"
    )
  else:
    c5.warning("📡 ĐANG QUÉT XAUUSD...")

  if scan_results:
    st.caption(
        "📡 **Radar quét Spot XAUUSD trực tiếp trên các khung thời gian:**"
    )
    st.dataframe(
        pd.DataFrame(scan_results).set_index("Khung").T,
        use_container_width=True,
    )

  if pos:
    p1, p2, p3, p4, p5 = st.columns(5)
    p1.info(
        f"**{pos['type']} (Khung {pos.get('tf', '5m')})** | Rủi ro"
        f" ${pos['risk']:.2f}"
    )
    p2.write(f"Vào: **${pos['entry']:,.2f}**")
    p3.write(f"TP: **${pos['tp']:,.2f}** | SL: **${pos['sl']:,.2f}**")
    p4.metric("Lãi/Lỗ Live", f"{fpnl:+,.2f} USD")
    if p5.button("❌ Chốt tay ngay", use_container_width=True):
      acc["bal"] = max(0.0, acc["bal"] + fpnl)
      acc["hist"].insert(
          0,
          {
              "Khung": pos.get("tf", "5m"),
              "Lệnh": pos["type"],
              "Vào": f"${pos['entry']:,.2f}",
              "Đóng": f"${live_price:,.2f}",
              "Lãi/Lỗ ($)": round(fpnl, 2),
              "Lý do": "Chốt tay 🖱️",
              "Giờ": now_s,
          },
      )
      acc["pos"] = None
      save_acc(acc)
      st.rerun()

  if acc["hist"]:
    hdf = pd.DataFrame(acc["hist"])
    wr = (len(hdf[hdf["Lãi/Lỗ ($)"] > 0]) / len(hdf)) * 100
    st.write(f"**Nhật ký giao dịch ({len(hdf)} lệnh) | Winrate: {wr:.1f}%**")
    st.dataframe(hdf, use_container_width=True)


@st.fragment(run_every=30)
def chart_panel():
  acc = load_acc()
  pos = acc["pos"]
  display_tf = pos["tf"] if (pos and "tf" in pos) else chart_tf
  recent, _, _ = analyze_xau_tf(display_tf)

  fig = go.Figure()
  fig.add_trace(
      go.Candlestick(
          x=recent.index,
          open=recent["HA_o"],
          high=recent["HA_h"],
          low=recent["HA_l"],
          close=recent["HA_c"],
          name="Heiken Ashi",
      )
  )
  fig.add_trace(
      go.Scatter(
          x=recent.index,
          y=recent["ema20"],
          line=dict(color="#FFD700", width=1.5),
          name="EMA 20",
      )
  )
  if pos:
    fig.add_hline(
        y=pos["entry"],
        line_color="white",
        annotation_text=f"ENTRY ({display_tf})",
    )
    fig.add_hline(
        y=pos["tp"], line_dash="dash", line_color="#00ff00", annotation_text="TP"
    )
    fig.add_hline(
        y=pos["sl"], line_dash="dash", line_color="#ff0000", annotation_text="SL"
    )

  fig.update_layout(
      height=420,
      xaxis_rangeslider_visible=False,
      template="plotly_dark",
      title=f"🥇 Biểu Đồ Vàng XAUUSD (Khung {display_tf})",
  )
  st.plotly_chart(fig, use_container_width=True)


live_xau_panel()
chart_panel()
