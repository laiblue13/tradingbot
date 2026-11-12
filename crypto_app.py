from datetime import datetime
import json, os
import ccxt
import numpy as np, pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import HistGradientBoostingClassifier
import streamlit as st

st.set_page_config(
    page_title="AI Crypto Pro Bot ($100)", layout="wide", page_icon="₿"
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

st.title("₿ Bot AI Giao Dịch Crypto Tự Động 24/7 (Vốn Demo $100)")
ACC_FILE = "crypto_demo_account.json"
ALL_COINS = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
ALL_TFS = ["5m", "15m", "30m", "1h"]


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
  st.header("🤖 Bảng Điều Khiển Crypto")
  auto_trade = st.toggle("🟢 BẬT TỰ ĐỘNG QUÉT & VÀO LỆNH", True)
  scan_coins = st.multiselect(
      "Các đồng Coin cho phép săn lệnh:", ALL_COINS, default=ALL_COINS
  )
  scan_tfs = st.multiselect(
      "Các khung thời gian quét:", ALL_TFS, default=["15m", "30m", "1h"]
  )
  chart_coin = st.selectbox("Coin hiển thị biểu đồ:", ALL_COINS, index=0)
  chart_tf = st.selectbox("Khung biểu đồ:", ALL_TFS, index=1)
  mtf_filter = st.checkbox("Lọc sóng lớn (1H EMA 50)", True)
  be_mode = st.checkbox("Dời SL hòa vốn khi lãi 1R", True)
  risk_pct = st.slider("Rủi ro mỗi lệnh (%):", 1.0, 5.0, 2.0, 0.5)
  max_loss_pct = st.slider("Khóa bot nếu lỗ ngày (%):", 3.0, 15.0, 6.0, 1.0)
  conf_th = st.slider("Độ tự tin AI tối thiểu (%):", 45, 75, 52, 2)
  if st.button("🔄 Reset ví Crypto về $100", use_container_width=True):
    if os.path.exists(ACC_FILE):
      os.remove(ACC_FILE)
    st.rerun()


@st.cache_data(ttl=60)
def analyze_crypto(symbol, tf):
  ex = ccxt.binance({"enableRateLimit": True})
  o1h = pd.DataFrame(
      ex.fetch_ohlcv(symbol, "1h", limit=100),
      columns=["t", "o", "h", "l", "c", "v"],
  )
  macro = (
      "UP"
      if float(o1h["c"].iloc[-1])
      > float(o1h["c"].ewm(span=50, adjust=False).mean().iloc[-1])
      else "DOWN"
  )

  tg_map = {"5m": 0.0020, "15m": 0.0035, "30m": 0.0050, "1h": 0.0070}
  raw = ex.fetch_ohlcv(symbol, tf, limit=1000)
  df = pd.DataFrame(
      raw, columns=["ts", "open", "high", "low", "close", "volume"]
  )
  df["ts"] = pd.to_datetime(df["ts"], unit="ms") + pd.Timedelta(hours=7)
  df.set_index("ts", inplace=True)
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
@st.fragment(run_every=2)
def live_crypto_panel():
  acc = load_acc()
  ex = ccxt.binance({"enableRateLimit": True})
  tickers = ex.fetch_tickers(ALL_COINS)
  live_prices = {c: float(tickers[c]["last"]) for c in ALL_COINS}
  now_s = datetime.now().strftime("%H:%M:%S")

  scan_list = []
  best_kèo = None

  for c in scan_coins:
    for t in scan_tfs:
      rec, probs, macro = analyze_crypto(c, t)
      sig, conf = int(np.argmax(probs)), float(np.max(probs) * 100)
      atr_val = float(rec["ATR"].iloc[-1])

      ok_dir = (
          (not mtf_filter)
          or (sig == 1 and macro == "UP")
          or (sig == 2 and macro == "DOWN")
      )
      st_txt = "🟡 Chờ"
      if sig == 1 and conf >= conf_th:
        st_txt = "🟢 LONG" if ok_dir else "⚠️ Ngược 1H"
      elif sig == 2 and conf >= conf_th:
        st_txt = "🔴 SHORT" if ok_dir else "⚠️ Ngược 1H"

      scan_list.append({
          "Cặp tiền": c,
          "Khung": t,
          "Sóng 1H": "🟢" if macro == "UP" else "🔴",
          "Tín hiệu AI": st_txt,
          "Độ tự tin": f"{conf:.1f}%",
      })

      if (sig == 1 or sig == 2) and conf >= conf_th and ok_dir:
        if best_kèo is None or conf > best_kèo["conf"]:
          best_kèo = {
              "coin": c,
              "tf": t,
              "sig": sig,
              "conf": conf,
              "atr": atr_val,
              "price": live_prices[c],
          }

  pos, fpnl = acc["pos"], 0.0
  if pos:
    c_price = live_prices[pos["coin"]]
    diff = (
        (c_price - pos["entry"])
        if pos["type"] == "LONG"
        else (pos["entry"] - c_price)
    )
    fpnl = (diff * pos["qty"]) - (pos["entry"] * pos["qty"] * 0.0004)
    if be_mode and not pos.get("be") and fpnl >= pos["risk"]:
      pos["sl"], pos["be"] = pos["entry"], True
      save_acc(acc)

    reason = None
    if pos["type"] == "LONG":
      if c_price >= pos["tp"]:
        reason = "Chốt lời 2R 🎯"
      elif c_price <= pos["sl"]:
        reason = "Hòa vốn 🛡️" if pos.get("be") else "Cắt lỗ 1R 🛑"
    else:
      if c_price <= pos["tp"]:
        reason = "Chốt lời 2R 🎯"
      elif c_price >= pos["sl"]:
        reason = "Hòa vốn 🛡️" if pos.get("be") else "Cắt lỗ 1R 🛑"

    if reason:
      acc["bal"] = max(0.0, acc["bal"] + fpnl)
      acc["hist"].insert(
          0,
          {
              "Coin": pos["coin"],
              "Khung": pos["tf"],
              "Lệnh": pos["type"],
              "Vào": f"${pos['entry']:,.2f}",
              "Đóng": f"${c_price:,.2f}",
              "Lãi/Lỗ ($)": round(fpnl, 2),
              "Lý do": reason,
              "Giờ": now_s,
          },
      )
      acc["pos"], pos, fpnl = None, None, 0.0
      save_acc(acc)

  day_pnl = (acc["bal"] + fpnl) - acc["day_start"]
  locked = day_pnl <= -(acc["day_start"] * (max_loss_pct / 100.0))

  if not pos and auto_trade and not locked and acc["bal"] >= 10.0 and best_kèo:
    risk_u = acc["bal"] * (risk_pct / 100.0)
    sl_d = best_kèo["atr"] * 1.2
    l_type = "LONG" if best_kèo["sig"] == 1 else "SHORT"
    p_in = best_kèo["price"]
    tp_p = round(p_in + sl_d * 2, 2) if l_type == "LONG" else round(p_in - sl_d * 2, 2)
    sl_p = round(p_in - sl_d, 2) if l_type == "LONG" else round(p_in + sl_d, 2)

    acc["pos"] = pos = {
        "coin": best_kèo["coin"],
        "tf": best_kèo["tf"],
        "type": l_type,
        "entry": p_in,
        "tp": tp_p,
        "sl": sl_p,
        "qty": risk_u / sl_d,
        "risk": risk_u,
        "be": False,
        "time": now_s,
    }
    save_acc(acc)

  eq = acc["bal"] + fpnl
  c1, c2, c3, c4, c5 = st.columns(5)
  c1.metric(f"₿ BTC ({now_s})", f"${live_prices['BTC/USDT']:,.1f}")
  c2.metric("Ξ ETH / ◎ SOL", f"${live_prices['ETH/USDT']:,.1f} | ${live_prices['SOL/USDT']:,.1f}")
  c3.metric("💰 Vốn khả dụng", f"${acc['bal']:,.2f}")
  c4.metric(
      "🏦 Tổng tài sản",
      f"${eq:,.2f}",
      f"{eq - acc['init']:+,.2f} USD (Ngày: {day_pnl:+,.2f}$)",
  )

  if locked:
    c5.error("🛑 KHÓA LỖ NGÀY")
  elif best_kèo:
    c5.success(
        f"🔥 {best_kèo['coin']} ({best_kèo['tf']}): {'LONG' if best_kèo['sig']==1 else 'SHORT'} ({best_kèo['conf']:.1f}%)"
    )
  else:
    c5.warning("📡 ĐANG QUÉT CRYPTO...")

  if pos:
    p1, p2, p3, p4, p5 = st.columns(5)
    p1.info(f"**{pos['type']} {pos['coin']} ({pos['tf']})** | Rủi ro ${pos['risk']:.2f}")
    p2.write(f"Vào: **${pos['entry']:,.2f}**")
    p3.write(f"TP: **${pos['tp']:,.2f}** | SL: **${pos['sl']:,.2f}**")
    p4.metric("Lãi/Lỗ Live", f"{fpnl:+,.2f} USD")
    if p5.button("❌ Chốt tay ngay", use_container_width=True):
      c_price = live_prices[pos["coin"]]
      acc["bal"] = max(0.0, acc["bal"] + fpnl)
      acc["hist"].insert(
          0,
          {
              "Coin": pos["coin"],
              "Khung": pos["tf"],
              "Lệnh": pos["type"],
              "Vào": f"${pos['entry']:,.2f}",
              "Đóng": f"${c_price:,.2f}",
              "Lãi/Lỗ ($)": round(fpnl, 2),
              "Lý do": "Chốt tay 🖱️",
              "Giờ": now_s,
          },
      )
      acc["pos"] = None
      save_acc(acc)
      st.rerun()

  if scan_list:
    st.caption("📡 **Radar quét Crypto đa khung thời gian (Đánh thuận chiều):**")
    st.dataframe(pd.DataFrame(scan_list), use_container_width=True, hide_index=True)

  if acc["hist"]:
    hdf = pd.DataFrame(acc["hist"])
    wr = (len(hdf[hdf["Lãi/Lỗ ($)"] > 0]) / len(hdf)) * 100
    st.write(f"**Nhật ký giao dịch Crypto ({len(hdf)} lệnh) | Winrate: {wr:.1f}%**")
    st.dataframe(hdf, use_container_width=True, hide_index=True)


@st.fragment(run_every=30)
def chart_panel():
  acc = load_acc()
  pos = acc["pos"]
  d_coin = pos["coin"] if pos else chart_coin
  d_tf = pos["tf"] if pos else chart_tf
  recent, _, _ = analyze_crypto(d_coin, d_tf)

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
    fig.add_hline(y=pos["entry"], line_color="white", annotation_text=f"ENTRY ({d_coin})")
    fig.add_hline(y=pos["tp"], line_dash="dash", line_color="#00ff00", annotation_text="TP")
    fig.add_hline(y=pos["sl"], line_dash="dash", line_color="#ff0000", annotation_text="SL")

  fig.update_layout(
      height=420,
      xaxis_rangeslider_visible=False,
      template="plotly_dark",
      title=f"₿ Biểu Đồ {d_coin} - Khung {d_tf}",
  )
  st.plotly_chart(fig, use_container_width=True)


live_crypto_panel()
chart_panel()
