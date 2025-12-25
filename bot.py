import os
import time
import asyncio
from dataclasses import dataclass

import pandas as pd
import yfinance as yf

from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

# ======================================================
# CONFIG
# ======================================================

load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = int(os.getenv("CHAT_ID", "0"))

# Ambicioso controlado (puedes ajustar en .env)
CONF_THRESHOLD = int(os.getenv("CONF_THRESHOLD", "68"))

SYMBOLS = {
    "BTC": "BTC-USD",
    "XAU": "GC=F",   # Oro futuros (Yahoo fiable)
    "XAG": "SI=F",   # Plata futuros (Yahoo fiable)
}

TIMEFRAME = os.getenv("TIMEFRAME", "5m")
LOOKBACK_PERIOD = os.getenv("LOOKBACK_PERIOD", "5d")

SIGNALS_CSV = "signals.csv"

# Antispam
ALERT_COOLDOWN_SECONDS = int(os.getenv("ALERT_COOLDOWN_SECONDS", str(20 * 60)))
_last_alert = {}

DATA_FAIL_COOLDOWN_SECONDS = int(os.getenv("DATA_FAIL_COOLDOWN_SECONDS", str(10 * 60)))
_last_data_fail = {}
_last_data_fail_notified = {}

# Indicadores
RSI_LEN = 14
EMA_FAST = 50
EMA_SLOW = 200
ATR_LEN = 14

# Filtro volatilidad (permisivo)
MIN_ATR_PCT = float(os.getenv("MIN_ATR_PCT", "0.0005"))  # 0.05%

# Risk/Reward
RR = float(os.getenv("RR", "1.3"))

# ======================================================
# MODELO
# ======================================================

@dataclass
class Signal:
    symbol: str
    action: str              # BUY / SELL / WAIT
    confidence: int
    price: float
    rsi: float
    trend: str               # UP / DOWN / SIDE
    atr: float
    lot: float
    sl: float
    tp: float
    setup: str               # TrendPullback / MeanReversion / Breakout / None
    why: str                 # por qué
    plan: str                # qué vigilar
    note: str = ""           # errores internos / data issues

# ======================================================
# INDICADORES
# ======================================================

def ema(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(span=n, adjust=False).mean()

def compute_rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(n).mean()
    avg_loss = loss.rolling(n).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def compute_atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(n).mean()

# ======================================================
# RIESGO
# ======================================================

def lot_from_conf(conf: int) -> float:
    if conf >= 80:
        return 1.0
    if conf >= 70:
        return 0.5
    if conf >= 60:
        return 0.1
    return 0.0

def sl_tp(price: float, atr: float, action: str) -> tuple[float, float]:
    if atr <= 0:
        return 0.0, 0.0
    risk = 1.0 * atr
    reward = RR * risk
    if action == "BUY":
        return price - risk, price + reward
    if action == "SELL":
        return price + risk, price - reward
    return 0.0, 0.0

# ======================================================
# DATA
# ======================================================

def fetch_ohlc(yahoo_symbol: str) -> pd.DataFrame:
    now = time.time()
    if now - _last_data_fail.get(yahoo_symbol, 0) < DATA_FAIL_COOLDOWN_SECONDS:
        return pd.DataFrame()

    df = yf.download(
        tickers=yahoo_symbol,
        period=LOOKBACK_PERIOD,
        interval=TIMEFRAME,
        auto_adjust=True,
        progress=False,
        threads=False,
    )

    if df is None or df.empty:
        _last_data_fail[yahoo_symbol] = now
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    df = df.rename(columns=lambda c: str(c).strip().title())
    needed = {"Open", "High", "Low", "Close"}
    if not needed.issubset(set(df.columns)):
        _last_data_fail[yahoo_symbol] = now
        return pd.DataFrame()

    return df.dropna()

# ======================================================
# CSV (FORMATO FIJO, SIN ROTURAS)
# ======================================================

CSV_COLUMNS = [
    "ts","symbol","yahoo_symbol","timeframe","source",
    "action","confidence","price","rsi","trend","atr",
    "lot","sl","tp","setup","why","plan","note"
]

def log_signal(sig: Signal, source: str):
    row = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "symbol": sig.symbol,
        "yahoo_symbol": SYMBOLS.get(sig.symbol, ""),
        "timeframe": TIMEFRAME,
        "source": source,
        "action": sig.action,
        "confidence": sig.confidence,
        "price": sig.price,
        "rsi": sig.rsi,
        "trend": sig.trend,
        "atr": sig.atr,
        "lot": sig.lot,
        "sl": sig.sl,
        "tp": sig.tp,
        "setup": sig.setup,
        "why": sig.why,
        "plan": sig.plan,
        "note": sig.note,
    }
    df = pd.DataFrame([row], columns=CSV_COLUMNS)
    df.to_csv(SIGNALS_CSV, mode="a", header=not os.path.exists(SIGNALS_CSV), index=False)

# ======================================================
# ANÁLISIS (MÁS “OPERABLE”)
# ======================================================

def trend_label(e50: float, e200: float) -> str:
    # margen para evitar falsas tendencias por ruido
    if e50 > e200 * 1.0005:
        return "UP"
    if e50 < e200 * 0.9995:
        return "DOWN"
    return "SIDE"

async def analyze_symbol(symbol: str) -> Signal:
    yahoo_symbol = SYMBOLS[symbol]
    df = fetch_ohlc(yahoo_symbol)

    if df.empty:
        return Signal(
            symbol=symbol, action="WAIT", confidence=0, price=0.0, rsi=0.0,
            trend="SIDE", atr=0.0, lot=0.0, sl=0.0, tp=0.0,
            setup="None",
            why="No hay datos suficientes para analizar.",
            plan=f"Esperar datos de {yahoo_symbol} en {TIMEFRAME}.",
            note=f"Sin datos para {yahoo_symbol} ({TIMEFRAME})."
        )

    close = df["Close"]
    price = float(close.iloc[-1])

    r = float(compute_rsi(close, RSI_LEN).iloc[-1])
    e50 = float(ema(close, EMA_FAST).iloc[-1])
    e200 = float(ema(close, EMA_SLOW).iloc[-1])
    a_series = compute_atr(df, ATR_LEN)
    a = float(a_series.iloc[-1]) if not pd.isna(a_series.iloc[-1]) else 0.0

    trend = trend_label(e50, e200)

    # Volatilidad mínima
    if a <= 0 or (a / price) < MIN_ATR_PCT:
        return Signal(
            symbol=symbol, action="WAIT", confidence=0, price=price, rsi=r,
            trend=trend, atr=a, lot=0.0, sl=0.0, tp=0.0,
            setup="None",
            why="Mercado lento (poca volatilidad).",
            plan="Esperar más movimiento o probar TF 15m.",
            note="Low ATR"
        )

    action = "WAIT"
    confidence = 55
    setup = "None"
    why = "No hay una entrada con ventaja clara."
    plan = "Esperar confluencia."

    # (1) Trend Pullback (más señales, riesgo controlado)
    if trend == "UP" and 40 <= r <= 52:
        action = "BUY"
        setup = "TrendPullback"
        confidence = 72 + (3 if r < 45 else 0)
        why = "Tendencia alcista + retroceso (pullback)."
        plan = "Confirmar rebote (vela verde o rechazo de mínimos) antes de entrar."

    elif trend == "DOWN" and 48 <= r <= 60:
        action = "SELL"
        setup = "TrendPullback"
        confidence = 72 + (3 if r > 55 else 0)
        why = "Tendencia bajista + pullback."
        plan = "Confirmar rechazo (vela roja o fallo en máximos) antes de entrar."

    # (2) Mean Reversion (extremos RSI) – más arriesgado
    elif r < 30:
        action = "BUY"
        setup = "MeanReversion"
        confidence = 62 + (6 if trend == "UP" else 0)
        why = "RSI muy bajo (sobreventa)."
        plan = "No entrar en caída libre: esperar rebote / giro visible."

    elif r > 70:
        action = "SELL"
        setup = "MeanReversion"
        confidence = 62 + (6 if trend == "DOWN" else 0)
        why = "RSI muy alto (sobrecompra)."
        plan = "No vender sin giro: esperar rechazo/vela de giro."

    # (3) Breakout moderado
    else:
        n = 20
        if len(df) > n + 1:
            recent_high = float(df["High"].iloc[-n-1:-1].max())
            recent_low = float(df["Low"].iloc[-n-1:-1].min())
            last_close = float(df["Close"].iloc[-1])

            if last_close > recent_high and trend == "UP" and r > 55:
                action = "BUY"
                setup = "Breakout"
                confidence = 75
                why = "Ruptura alcista + tendencia a favor."
                plan = "Evitar entrar tarde: mejor esperar pullback o cierre fuerte."

            elif last_close < recent_low and trend == "DOWN" and r < 45:
                action = "SELL"
                setup = "Breakout"
                confidence = 75
                why = "Ruptura bajista + tendencia a favor."
                plan = "Confirmar con cierre; evita perseguir el precio."

    # SL/TP y lote si hay operación
    lot = 0.0
    sl = tp = 0.0
    if action != "WAIT":
        lot = lot_from_conf(confidence)
        sl, tp = sl_tp(price, a, action)

    # Si WAIT, dar “qué falta” directo
    if action == "WAIT":
        if trend == "UP":
            plan = "Preferencia BUY. Señal si RSI baja a 40–52 (pullback) o rompe máximos con fuerza."
        elif trend == "DOWN":
            plan = "Preferencia SELL. Señal si RSI sube a 48–60 (pullback) o rompe mínimos con fuerza."
        else:
            plan = "Lateral. Esperar ruptura o extremos (RSI<30 o RSI>70)."

    return Signal(
        symbol=symbol,
        action=action,
        confidence=int(min(90, max(0, confidence))),
        price=price,
        rsi=r,
        trend=trend,
        atr=a,
        lot=lot,
        sl=sl,
        tp=tp,
        setup=setup,
        why=why,
        plan=plan,
        note="",
    )

# ======================================================
# MENSAJES (HUMANOS)
# ======================================================

def trend_to_text(t: str) -> str:
    return {"UP": "ALCISTA", "DOWN": "BAJISTA", "SIDE": "LATERAL"}.get(t, "LATERAL")

def format_signal(sig: Signal) -> str:
    if sig.confidence == 0 and sig.price == 0.0:
        return (
            f"📊 {sig.symbol}\n"
            f"Estado: SIN DATOS\n"
            f"Motivo: {sig.note}\n"
            f"Plan: {sig.plan}"
        )

    header = (
        f"📊 {sig.symbol}\n"
        f"Tendencia: {trend_to_text(sig.trend)} | RSI: {sig.rsi:.1f}\n"
        f"Precio: {sig.price:.2f}\n"
    )

    if sig.action == "WAIT":
        return (
            header +
            f"Decisión: ESPERAR\n"
            f"Por qué: {sig.why}\n"
            f"Qué vigilar: {sig.plan}"
        )

    return (
        header +
        f"Decisión: {sig.action} ({sig.setup})\n"
        f"Confianza: {sig.confidence}% | Lote sugerido: {sig.lot}\n"
        f"SL: {sig.sl:.2f} | TP: {sig.tp:.2f} (RR≈{RR})\n"
        f"Por qué: {sig.why}\n"
        f"Plan: {sig.plan}"
    )

# ======================================================
# ALERTAS
# ======================================================

async def alert_tick(app: Application):
    for sym in SYMBOLS.keys():
        sig = await analyze_symbol(sym)
        now = time.time()

        # Aviso de data (1 vez por cooldown)
        if sig.confidence == 0 and sig.price == 0.0:
            last_note = _last_data_fail_notified.get(sym, 0)
            if now - last_note >= DATA_FAIL_COOLDOWN_SECONDS:
                await app.bot.send_message(chat_id=CHAT_ID, text=format_signal(sig))
                log_signal(sig, "alert")
                _last_data_fail_notified[sym] = now
            continue

        # Alertas operativas
        if sig.action != "WAIT" and sig.confidence >= CONF_THRESHOLD:
            last = _last_alert.get(sym, 0)
            if now - last >= ALERT_COOLDOWN_SECONDS:
                await app.bot.send_message(chat_id=CHAT_ID, text=format_signal(sig))
                log_signal(sig, "alert")
                _last_alert[sym] = now

async def on_start(app: Application):
    await app.bot.send_message(chat_id=CHAT_ID, text="✅ Bot iniciado (estable + listo para subir)")
    app.job_queue.run_repeating(
        lambda ctx: asyncio.create_task(alert_tick(ctx.application)),
        interval=60,
        first=5,
    )

# ======================================================
# COMANDOS (ACEPTA MAYÚSCULAS)
# ======================================================

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.effective_message.reply_text(
        "👋 Bot listo.\nComandos:\n"
        "/status\n/btc\n/xau\n/xag\n/stats\n"
        f"Umbral alertas: {CONF_THRESHOLD}%"
    )

async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.effective_message.reply_text("✅ Bot activo y funcionando")

async def cmd_asset(update: Update, context: ContextTypes.DEFAULT_TYPE, sym: str):
    sig = await analyze_symbol(sym)
    log_signal(sig, "command")
    await update.effective_message.reply_text(format_signal(sig))

async def cmd_btc(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await cmd_asset(update, context, "BTC")

async def cmd_xau(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await cmd_asset(update, context, "XAU")

async def cmd_xag(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await cmd_asset(update, context, "XAG")

async def cmd_stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not os.path.exists(SIGNALS_CSV):
        await update.effective_message.reply_text("Aún no hay historial. Usa /btc /xau /xag.")
        return

    # lectura tolerante (si alguna línea quedara corrupta)
    try:
        df = pd.read_csv(SIGNALS_CSV, on_bad_lines="skip")
    except TypeError:
        df = pd.read_csv(SIGNALS_CSV, engine="python")

    if df.empty:
        await update.effective_message.reply_text("Historial vacío.")
        return

    total = len(df)
    by_sym = df.groupby("symbol")["action"].value_counts().unstack(fill_value=0)
    conf_mean = df[df["price"] > 0].groupby("symbol")["confidence"].mean().round(1)

    msg = [f"📈 STATS (registros: {total})"]
    for sym in sorted(df["symbol"].unique()):
        row = by_sym.loc[sym] if sym in by_sym.index else None
        if row is not None:
            msg.append(f"- {sym}: BUY={int(row.get('BUY',0))} | SELL={int(row.get('SELL',0))} | WAIT={int(row.get('WAIT',0))}")
        if sym in conf_mean.index:
            msg.append(f"  Confianza media: {conf_mean[sym]}%")

    await update.effective_message.reply_text("\n".join(msg))

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    msg = f"⚠️ Error interno: {context.error}"
    print(msg)
    try:
        if CHAT_ID:
            await context.bot.send_message(chat_id=CHAT_ID, text=msg)
    except Exception:
        pass

# ======================================================
# MAIN
# ======================================================

def main():
    if not BOT_TOKEN:
        raise RuntimeError("Falta BOT_TOKEN en .env")
    if CHAT_ID == 0:
        raise RuntimeError("Falta CHAT_ID en .env")

    app = Application.builder().token(BOT_TOKEN).post_init(on_start).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("btc", cmd_btc))
    app.add_handler(CommandHandler("BTC", cmd_btc))  # por si escribes /BTC
    app.add_handler(CommandHandler("xau", cmd_xau))
    app.add_handler(CommandHandler("XAU", cmd_xau))
    app.add_handler(CommandHandler("xag", cmd_xag))
    app.add_handler(CommandHandler("XAG", cmd_xag))
    app.add_handler(CommandHandler("stats", cmd_stats))

    app.add_error_handler(error_handler)
    app.run_polling(allowed_updates=Update.ALL_TYPES)

# ======================
# MINI SERVIDOR HTTP (Render Free)
# ======================
from flask import Flask
import threading
import os

def run_web():
    app = Flask(__name__)

    @app.route("/")
    def home():
        return "Bot activo OK", 200

    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

threading.Thread(target=run_web, daemon=True).start()

if __name__ == "__main__":
    main()
