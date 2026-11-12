from datetime import datetime
import os
import time
import ccxt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

# ==========================================
# 1. CẤU HÌNH API BINANCE & AN TOÀN VỐN $100
# ==========================================
API_KEY = os.getenv("BINANCE_API_KEY", "DÁN_API_KEY_BINANCE_CỦA_BẠN")
SECRET_KEY = os.getenv("BINANCE_SECRET_KEY", "DÁN_SECRET_KEY_BINANCE_CỦA_BẠN")

USE_TESTNET = (
    True  # True = Chạy trên Binance Futures Testnet | False = Tiền thật
)
TIMEFRAME = "15m"  # Khung thời gian đánh chính
RISK_PER_TRADE = 2.0  # Rủi ro tối đa 2 USDT/lệnh (2% vốn $100)
MAX_LEVERAGE = 5  # Đòn bẩy an toàn x5
MAX_POSITION_USD = 400.0  # Giá trị vị thế tối đa (Không vượt quá 80% sức mua x5)
CONF_THRESHOLD = 58.0  # Độ tự tin AI tối thiểu (%)

exchange = ccxt.binance({
    "apiKey": API_KEY,
    "secret": SECRET_KEY,
    "enableRateLimit": True,
    "options": {
        "defaultType": "future",  # Giao dịch USDⓈ-M Futures
        "adjustForTimeDifference": True,
    },
})

if USE_TESTNET:
  exchange.set_sandbox_mode(True)


# ==========================================
# 2. LỌC TOP 15 COIN THANH KHOẢN CAO NHẤT BINANCE
# ==========================================
def get_top_binance_coins(limit=15):
  tickers = exchange.fetch_tickers()
  usdt_futures = []
  for sym, data in tickers.items():
    if sym.endswith("/USDT:USDT") and data.get("quoteVolume"):
      # Trên Binance, quoteVolume đã tính sẵn bằng USDT
      vol_usdt = float(data["quoteVolume"])
      usdt_futures.append((sym, vol_usdt))

  # Sắp xếp giảm dần theo khối lượng giao dịch 24h (phần tử thứ 2 của tuple)
  usdt_futures.sort(key=lambda item: item[1], reverse=True)
  # Chỉ lấy tên cặp giao dịch (phần tử đầu tiên của tuple)
  return [item[0] for item in usdt_futures[:limit]]
# ==========================================
# 3. BỘ NÃO AI PHÂN TÍCH XU HƯỚNG (ĐÃ FIX REPAINT)
# ==========================================
def analyze_coin_ai(symbol, tf):
  # 3.1 Phân tích xu hướng khung lớn 1H (dùng nến đã đóng iloc[-2])
  raw_1h = exchange.fetch_ohlcv(symbol, "1h", limit=100)
  df_1h = pd.DataFrame(raw_1h, columns=["ts", "o", "h", "l", "c", "v"])
  ema50_1h = df_1h["c"].ewm(span=50, adjust=False).mean()
  macro = (
      "UP"
      if float(df_1h["c"].iloc[-2]) > float(ema50_1h.iloc[-2])
      else "DOWN"
  )

  # 3.2 Phân tích khung giao dịch chính (15m)
  raw = exchange.fetch_ohlcv(symbol, tf, limit=250)
  df = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "volume"])
  df["hr"] = pd.to_datetime(df["ts"], unit="ms").dt.hour

  # Tính nến Heikin Ashi
  df["HA_c"] = (df["open"] + df["high"] + df["low"] + df["close"]) / 4
  first_o = float(df["open"].iloc)
  first_c = float(df["close"].iloc)
  ha_o = [(first_o + first_c) / 2]
  hac_list = df["HA_c"].tolist()
  for i in range(1, len(df)):
    ha_o.append((ha_o[i - 1] + hac_list[i - 1]) / 2)
  df["HA_o"] = ha_o
  df["HA_h"] = df[["high", "HA_o", "HA_c"]].max(axis=1)
  df["HA_l"] = df[["low", "HA_o", "HA_c"]].min(axis=1)

  # Đặc trưng hình thái nến
  df["col"] = np.where(df["HA_c"] > df["HA_o"], 1, -1)
  df["body"] = (df["HA_c"] - df["HA_o"]) / df["HA_o"]
  df["uw"] = (df["HA_h"] - df[["HA_o", "HA_c"]].max(axis=1)) / df["HA_o"]
  df["lw"] = (df[["HA_o", "HA_c"]].min(axis=1) - df["HA_l"]) / df["HA_o"]

  # Chỉ báo RSI và EMA20
  d = df["HA_c"].diff()
  up_ema = d.where(d > 0, 0).ewm(alpha=1 / 14, adjust=False).mean()
  down_ema = (-d.where(d < 0, 0)).ewm(alpha=1 / 14, adjust=False).mean()
  df["rsi"] = 100 - (100 / (1 + up_ema / down_ema.replace(0, np.nan)))
  df["ema20"] = df["HA_c"].ewm(span=20, adjust=False).mean()
  df["dist_ema"] = (df["HA_c"] - df["ema20"]) / df["ema20"]

  # Tính biên độ dao động thực tế (ATR)
  tr = pd.concat(
      [
          df["high"] - df["low"],
          (df["high"] - df["close"].shift(1)).abs(),
          (df["low"] - df["close"].shift(1)).abs(),
      ],
      axis=1,
  ).max(axis=1)
  df["ATR"] = tr.rolling(14).mean()

  # Tạo nhãn dự đoán 3 nến tiếp theo (loại bỏ rò rỉ dữ liệu tương lai)
  ret = df["close"].shift(-3) / df["close"] - 1
  df["target"] = np.where(ret > 0, 1, 0)

  # Chỉ dùng các dòng đã hoàn tất dữ liệu tương lai để train
  train_df = df.iloc[:-4].dropna().copy()

  feats = ["hr", "col", "body", "uw", "lw", "rsi", "dist_ema"]
  model = HistGradientBoostingClassifier(
      max_iter=120, learning_rate=0.05, max_depth=4, random_state=42
  )
  model.fit(train_df[feats], train_df["target"])

  # Dự đoán trên cây nến VỪA ĐÓNG CỬA gần nhất (iloc[-2:-1]) để tránh repaint
  probs = model.predict_proba(df[feats].iloc[-2:-1])
  p_short, p_long = probs.tolist()

  # Lấy giá hiện tại và ATR của nến đã đóng
  curr_price = float(df["close"].iloc[-1])
  atr_val = float(df["ATR"].iloc[-2])

  return p_short * 100, p_long * 100, macro, curr_price, atr_val
# ==========================================
# 4. ĐẶT LỆNH AN TOÀN LÊN BINANCE FUTURES (ISOLATED x5)
# ==========================================
def execute_binance_order(symbol, side, price, atr):
  try:
    market = exchange.market(symbol)

    # 1. Cài đặt chế độ Isolated (bỏ qua nếu đã cài sẵn)
    try:
      exchange.set_margin_mode("isolated", symbol)
    except Exception:
      pass

    # 2. Cài đặt đòn bẩy x5
    exchange.set_leverage(MAX_LEVERAGE, symbol)

    # 3. Kiểm tra an toàn thanh lý (Tránh SL xa hơn giá cháy của đòn bẩy x5)
    sl_dist = atr * 1.5
    sl_pct = sl_dist / price
    if sl_pct > (0.8 / MAX_LEVERAGE):  # Tối đa 16% cho đòn bẩy x5
      print(
          f"⚠️ Bỏ qua {symbol}: Biên độ SL quá rộng ({sl_pct*100:.1f}%), rủi ro"
          " thanh lý trước khi chạm SL."
      )
      return None

    # 4. Tính khối lượng coin theo rủi ro $2
    raw_qty = RISK_PER_TRADE / sl_dist
    if raw_qty * price > MAX_POSITION_USD:
      raw_qty = MAX_POSITION_USD / price

    qty = float(exchange.amount_to_precision(symbol, raw_qty))
    min_qty = float(market["limits"]["amount"]["min"] or 0.0)
    min_cost = float(
        market["limits"]["cost"]["min"] or 5.0
    )  # Binance tối thiểu 5 USDT

    if qty < min_qty or (qty * price) < min_cost:
      print(
          f"⚠️ Bỏ qua {symbol}: Giá trị lệnh ({qty * price:.2f} USDT) nhỏ hơn"
          f" mức tối thiểu ({min_cost} USDT)."
      )
      return None

    # 5. Tính toán mức giá TP (R:R = 1:2) và SL
    close_side = "sell" if side == "BUY" else "buy"
    if side == "BUY":
      tp_price = float(exchange.price_to_precision(symbol, price + sl_dist * 2))
      sl_price = float(exchange.price_to_precision(symbol, price - sl_dist))
    else:
      tp_price = float(exchange.price_to_precision(symbol, price - sl_dist * 2))
      sl_price = float(exchange.price_to_precision(symbol, price + sl_dist))

    # 6. Gửi lệnh Market vào vị thế chính
    exchange.create_market_order(symbol, side.lower(), qty)

    # 7. Gửi 2 lệnh điều kiện Take Profit và Stop Loss độc lập (bắt buộc reduceOnly=True)
    exchange.create_order(
        symbol=symbol,
        type="TAKE_PROFIT_MARKET",
        side=close_side,
        amount=qty,
        params={
            "stopPrice": tp_price,
            "reduceOnly": True,
            "workingType": "MARK_PRICE",  # Dùng Mark Price để tránh râu nến quét SL ảo
        },
    )
    exchange.create_order(
        symbol=symbol,
        type="STOP_MARKET",
        side=close_side,
        amount=qty,
        params={
            "stopPrice": sl_price,
            "reduceOnly": True,
            "workingType": "MARK_PRICE",
        },
    )

    notional_val = qty * price
    print(
        f"✅ ĐÃ KHỚP LỆNH BINANCE: {side} {symbol} | {qty} coin"
        f" (~{notional_val:.1f} USDT) | Giá: {price}"
    )
    print(
        f"🛡️ Đã đặt lệnh treo TP/SL cứng -> TP: {tp_price} | SL: {sl_price}"
    )
    return symbol

  except Exception as e:
    print(f"❌ Lỗi đặt lệnh Binance {symbol}: {e}")
    return None


# ==========================================
# 5. VÒNG LẶP VẬN HÀNH 24/7 TRÊN BINANCE
# ==========================================
print("🚀 KHỞI ĐỘNG BOT BINANCE FUTURES (CẤU HÌNH AN TOÀN VỐN $100)...")
exchange.load_markets()
last_traded_symbol = None

while True:
  try:
    now = datetime.now()
    now_str = now.strftime("%H:%M:%S")

    # Kiểm tra các vị thế đang mở trên Binance
    positions = exchange.fetch_positions()
    active_pos = [p for p in positions if abs(float(p.get("contracts") or 0)) > 0]

    if len(active_pos) > 0:
      pos = active_pos[0]
      last_traded_symbol = pos["symbol"]
      pnl = float(pos.get("unrealizedPnl") or 0.0)
      print(
          f"[{now_str}] ⏳ Binance đang giữ lệnh {pos['symbol']}"
          f" ({pos['side'].upper()}) | Lãi/Lỗ: {pnl:+.2f} USDT"
      )
      time.sleep(20)
      continue

    # Nếu không còn vị thế mở, dọn dẹp lệnh TP/SL thừa của kèo trước (nếu có)
    if last_traded_symbol:
      try:
        exchange.cancel_all_orders(last_traded_symbol)
        print(
            f"[{now_str}] 🧹 Đã dọn sạch lệnh TP/SL thừa của"
            f" {last_traded_symbol}."
        )
      except Exception:
        pass
      last_traded_symbol = None

    print(
        f"\n[{now_str}] 🔍 Đang quét Top 15 Coin sôi động nhất trên Binance"
        " Futures..."
    )
    top_coins = get_top_binance_coins(15)
    best_trade = None

    for coin in top_coins:
      p_short, p_long, macro, price, atr = analyze_coin_ai(coin, TIMEFRAME)
      if p_long >= CONF_THRESHOLD and macro == "UP":
        if best_trade is None or p_long > best_trade["conf"]:
          best_trade = {
              "coin": coin,
              "side": "BUY",
              "conf": p_long,
              "price": price,
              "atr": atr,
          }
      elif p_short >= CONF_THRESHOLD and macro == "DOWN":
        if best_trade is None or p_short > best_trade["conf"]:
          best_trade = {
              "coin": coin,
              "side": "SELL",
              "conf": p_short,
              "price": price,
              "atr": atr,
          }

    if best_trade:
      print(
          f"🔥 TÍN HIỆU CHUẨN BINANCE: {best_trade['coin']} ->"
          f" {best_trade['side']} (Tự tin: {best_trade['conf']:.1f}%)"
      )
      last_traded_symbol = execute_binance_order(
          best_trade["coin"],
          best_trade["side"],
          best_trade["price"],
          best_trade["atr"],
      )
    else:
      print(
          f"[{now_str}] 🟡 Chưa có kèo đạt chuẩn {CONF_THRESHOLD}% thuận sóng"
          " 1H trên Binance."
      )

    # Nghỉ 60 giây trước lần quét tiếp theo để tối ưu CPU và tránh dính Rate Limit Binance
    time.sleep(60)

  except Exception as e:
    print(f"⚠️ Lỗi kết nối Binance tạm thời: {e}")
    time.sleep(15)
