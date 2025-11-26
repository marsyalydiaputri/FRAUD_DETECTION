# app.py
import os
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
import duckdb
import plotly.express as px
import numpy as np

# ---------- Page setup ----------
st.set_page_config(
    page_title="📊 AI Fraud Detection",
    page_icon="📊",
    layout="wide"
)

st.title("📊 AI Fraud Detection")
st.caption("Prototype v2.0 - Rule-based & AI Commentary + Chat Mode")

# ---------- Load API Keys ----------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Try import groq (optional)
has_groq = False
groq_client = None
if GROQ_API_KEY:
    try:
        import groq  # type: ignore
        groq_client = groq.Client(api_key=GROQ_API_KEY)
        has_groq = True
    except Exception:
        has_groq = False
        groq_client = None

if not GROQ_API_KEY:
    st.sidebar.warning("⚠️ AI Commentary tidak aktif (GROQ_API_KEY tidak ditemukan).")

# ---------- Helpers ----------
def detect_outliers_iqr(series: pd.Series) -> pd.Series:
    if series.dropna().empty:
        return pd.Series([False] * len(series), index=series.index)
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return (series < lower) | (series > upper)

def detect_outliers_zscore(series: pd.Series, thresh=3.5) -> pd.Series:
    if series.dropna().empty or series.std() == 0:
        return pd.Series([False] * len(series), index=series.index)
    z = (series - series.mean()) / series.std()
    return z.abs() > thresh

def generate_ai_commentary_vendor(agg_vendor_df: pd.DataFrame) -> str:
    """
    AI commentary in Indonesian focused on fraud detection using vendor-level aggregation.
    Uses GROQ if available; otherwise returns notice.
    """
    if not GROQ_API_KEY:
        return "⚠️ AI Commentary tidak aktif (GROQ_API_KEY tidak ditemukan)."

    if not has_groq or groq_client is None:
        return "⚠️ AI Commentary tidak aktif karena library `groq` tidak tersedia di environment."

    prompt_lines = [
        "Anda adalah analis risiko/penipuan (fraud analyst).",
        "Berdasarkan ringkasan total transaksi per Vendor di bawah, berikan analisis singkat dalam bahasa Indonesia:",
        "- Sebutkan Vendor dominan (nilai dan persentase dari total transaksi).",
        "- Sebutkan Vendor yang perlu mendapat perhatian (potensi anomali/pola tidak wajar).",
        "- Berikan 2-3 rekomendasi investigasi lanjutan atau mitigasi risiko.",
        "",
        "Data ringkasan (Vendor - TotalAmount):"
    ]
    total_all = agg_vendor_df["TotalAmount"].sum() if not agg_vendor_df.empty else 0
    for _, row in agg_vendor_df.iterrows():
        pct = (row["TotalAmount"] / total_all * 100) if total_all else 0
        prompt_lines.append(f"- {row['Vendor']}: {row['TotalAmount']:.2f} ( {pct:.1f}% )")
    prompt = "\n".join(prompt_lines)

    try:
        response = groq_client.generate(
            model="llama-3.3-70b-versatile",
            prompt=prompt,
            max_tokens=500
        )
        if isinstance(response, dict) and "text" in response:
            return response["text"].strip()
        return getattr(response, "text", str(response)).strip()
    except Exception as e:
        return f"⚠️ Gagal menghasilkan AI commentary: {e}"

def simple_ai_reply_chat(user_message: str, context: dict) -> str:
    """
    Fallback local reply when GROQ not available. Uses simple heuristics.
    context: dict with keys 'vendor_agg', 'flagged_df'
    """
    vendor_agg = context.get("vendor_agg", pd.DataFrame())
    flagged = context.get("flagged_df", pd.DataFrame())

    if vendor_agg is None or vendor_agg.empty:
        return "Saya belum menerima ringkasan transaksi per vendor. Silakan unggah data transaksi yang memiliki kolom 'Amount' dan 'Vendor'."

    top = vendor_agg.iloc[0]
    total = vendor_agg["TotalAmount"].sum()
    top_share = (top["TotalAmount"] / total * 100) if total != 0 else 0
    reply = f"Vendor dominan: **{top['Vendor']}** dengan total {top['TotalAmount']:,.2f} ({top_share:.1f}% dari total transaksi).\n"

    lower = user_message.lower()
    if any(k in lower for k in ["fraud", "penipuan", "anomali", "anomali", "curang", "mencurigakan"]):
        n_flagged = len(flagged) if flagged is not None else 0
        reply += f"Ada **{n_flagged}** transaksi yang ter-flag sebagai outlier berdasarkan heuristik. Prioritaskan pemeriksaan EmployeeID, Account, InvoiceNumber, serta waktu transaksi."
    elif "mengapa" in lower or "kenapa" in lower:
        reply += "Perbedaan antar vendor dapat disebabkan ukuran vendor, frekuensi transaksi, atau transaksi luar biasa (spike). Cek distribusi Amount per vendor."
    else:
        reply += "Apakah Anda ingin saya tampilkan contoh transaksi yang di-flag atau ringkasan per Account/EmployeeID?"

    return reply

# ---------- Data upload ----------
st.sidebar.header("Data Upload")
uploaded_file = st.sidebar.file_uploader(
    "Unggah file CSV atau Excel (kolom minimal: TransactionID, Date, Time, Vendor, Amount, Account, EmployeeID, InvoiceNumber, Description)",
    type=["csv", "xlsx", "xls"]
)

df = None
if uploaded_file is not None:
    try:
        if uploaded_file.name.lower().endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        st.success(f"File '{uploaded_file.name}' berhasil diunggah. Baris: {len(df)}")
    except Exception as e:
        st.error(f"Gagal membaca file: {e}")
        df = None

# Preview
st.subheader("Preview Data Transaksi")
if df is None:
    st.info("Belum ada data. Silakan unggah file CSV/Excel di sidebar.")
else:
    st.dataframe(df.head(200))

# Prepare Amount column
if df is not None:
    if "Amount" not in df.columns and "Sales" in df.columns:
        # fallback if user provided Sales
        df["Amount"] = pd.to_numeric(df["Sales"], errors="coerce")
    if "Amount" in df.columns:
        df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce").fillna(0)
    else:
        st.warning("⚠️ Data tidak memiliki kolom 'Amount'. Analisis fraud membutuhkan kolom 'Amount'.")

# ---------- Aggregations with DuckDB (fraud-relevant) ----------
st.subheader("Agregasi Transaksi (DuckDB) — Fokus Fraud Detection")

vendor_agg = pd.DataFrame()
account_agg = pd.DataFrame()
employee_count = pd.DataFrame()
time_pattern = pd.DataFrame()

if df is not None and "Amount" in df.columns and "Vendor" in df.columns:
    try:
        con = duckdb.connect(database=":memory:")
        con.register("txn", df)
        vendor_agg = con.execute("""
            SELECT Vendor, SUM(Amount) AS TotalAmount, COUNT(*) AS TxnCount
            FROM txn
            GROUP BY Vendor
            ORDER BY TotalAmount DESC
        """).fetchdf()
        account_agg = con.execute("""
            SELECT Account, SUM(Amount) AS TotalAmount, COUNT(*) AS TxnCount
            FROM txn
            GROUP BY Account
            ORDER BY TotalAmount DESC
        """).fetchdf()
        if "EmployeeID" in df.columns:
            employee_count = con.execute("""
                SELECT EmployeeID, COUNT(*) AS TxnCount, SUM(Amount) AS TotalAmount
                FROM txn
                GROUP BY EmployeeID
                ORDER BY TxnCount DESC
            """).fetchdf()
        # time pattern: by hour if Time available, else by Date
        if "Time" in df.columns:
            # try extract hour
            df_temp = df.copy()
            try:
                df_temp["Hour"] = pd.to_datetime(df_temp["Time"], errors="coerce").dt.hour
                con.register("tmp_time", df_temp)
                time_pattern = con.execute("""
                    SELECT Hour, COUNT(*) AS TxnCount, SUM(Amount) AS TotalAmount
                    FROM tmp_time
                    GROUP BY Hour
                    ORDER BY Hour
                """).fetchdf()
            except Exception:
                time_pattern = pd.DataFrame()
        else:
            time_pattern = pd.DataFrame()
        st.write("Tabel agregasi: Total Amount per Vendor")
        st.dataframe(vendor_agg.head(200))
    except Exception as e:
        st.error(f"Terjadi error saat agregasi dengan DuckDB: {e}")
else:
    st.warning("⚠️ Untuk agregasi per Vendor diperlukan kolom 'Vendor' dan 'Amount'.")

# ---------- Outlier detection (rule-based) ----------
st.subheader("Deteksi Transaksi Mencurigakan (Heuristik Rule-based)")
flagged_df = pd.DataFrame()
if df is not None and "Amount" in df.columns:
    try:
        df_num = df.copy()
        df_num["Amount"] = pd.to_numeric(df_num["Amount"], errors="coerce").fillna(0)
        mask_iqr = detect_outliers_iqr(df_num["Amount"])
        mask_z = detect_outliers_zscore(df_num["Amount"], thresh=3.5)
        mean = df_num["Amount"].mean()
        std = df_num["Amount"].std() if not np.isnan(df_num["Amount"].std()) else 0
        mask_extreme = df_num["Amount"] > (mean + 4 * std) if std and std > 0 else pd.Series([False]*len(df_num), index=df_num.index)
        combined_mask = mask_iqr | mask_z | mask_extreme
        flagged_df = df_num[combined_mask].copy()
        st.markdown(f"- Ditemukan **{len(flagged_df)}** transaksi yang di-flag sebagai outlier (heuristik).")
        if not flagged_df.empty:
            st.dataframe(flagged_df.head(200))
            st.markdown("Catatan: Ini hanya heuristik awal. Untuk investigasi, korelasikan dengan EmployeeID, Account, InvoiceNumber, dan waktu transaksi.")
    except Exception as e:
        st.error(f"Gagal mendeteksi outlier: {e}")
else:
    st.info("Deteksi outlier membutuhkan kolom 'Amount'.")

# ---------- Visualization (Plotly) ----------
st.subheader("Visualisasi: Total Amount by Vendor")
if vendor_agg is None or vendor_agg.empty:
    st.info("Grafik tidak tersedia — pastikan data memiliki kolom 'Vendor' dan 'Amount' dan telah diunggah.")
else:
    try:
        fig = px.bar(vendor_agg, x="Vendor", y="TotalAmount", title="Total Amount by Vendor", labels={"TotalAmount": "Total Amount"})
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Gagal membuat grafik: {e}")

# Additional visualizations
st.subheader("Visualisasi Tambahan")
cols_vis = st.columns(2)
with cols_vis[0]:
    st.markdown("**Top Accounts by Total Amount**")
    if not account_agg.empty:
        fig2 = px.bar(account_agg.head(20), x="Account", y="TotalAmount", title="Total Amount by Account", labels={"TotalAmount": "Total Amount"})
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Tidak ada data Account atau hasil agregasi kosong.")
with cols_vis[1]:
    st.markdown("**Transaction Pattern by Hour**")
    if not time_pattern.empty:
        fig3 = px.line(time_pattern, x="Hour", y="TxnCount", title="Transaction Count by Hour", labels={"TxnCount": "Transaction Count"})
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("Kolom 'Time' tidak tersedia atau tidak dapat diparse ke jam.")

# ---------- Rule-based Commentary (Fraud-focused) ----------
st.subheader("Rule-based Commentary (Fraud-focused)")
if vendor_agg is None or vendor_agg.empty:
    st.info("Rule-based commentary tidak tersedia karena agregasi per Vendor kosong.")
else:
    try:
        top_vendor = vendor_agg.iloc[0]
        low_vendor = vendor_agg.iloc[-1]
        gap = top_vendor["TotalAmount"] - low_vendor["TotalAmount"]
        st.markdown(f"- **Vendor dengan total transaksi tertinggi:** {top_vendor['Vendor']} — {top_vendor['TotalAmount']:,.2f}")
        st.markdown(f"- **Vendor dengan total transaksi terendah:** {low_vendor['Vendor']} — {low_vendor['TotalAmount']:,.2f}")
        st.markdown(f"- **Gap (tertinggi - terendah):** {gap:,.2f}")
        st.markdown("> Analisis cepat: konsentrasi transaksi pada beberapa vendor bisa menandakan risiko (mis. collusion, markup, atau vendor palsu). Periksa transaksi besar dan pola berulang.")
    except Exception as e:
        st.error(f"Gagal membuat rule-based commentary: {e}")

# ---------- AI Commentary (opsional) ----------
st.subheader("AI Commentary (opsional — fokus fraud detection)")
if not GROQ_API_KEY:
    st.info("⚠️ AI Commentary tidak aktif.")
else:
    if vendor_agg is None or vendor_agg.empty:
        st.info("AI Commentary membutuhkan ringkasan 'Vendor' & 'TotalAmount'.")
    else:
        with st.expander("Tampilkan AI Commentary"):
            ai_text = generate_ai_commentary_vendor(vendor_agg)
            if ai_text:
                if ai_text.startswith("⚠️"):
                    st.warning(ai_text)
                else:
                    st.markdown(ai_text)

# ---------- AI Chat Mode ----------
st.subheader("AI Chat Mode — Interaktif (Analis Fraud)")
st.markdown("AI bertindak sebagai analis fraud/risk. Gunakan chat untuk bertanya tentang potensi penipuan, transaksi outlier, atau rekomendasi investigasi. Riwayat disimpan di session_state.")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "vendor_agg_cache" not in st.session_state:
    st.session_state["vendor_agg_cache"] = vendor_agg.copy() if not vendor_agg.empty else pd.DataFrame()
if "flagged_cache" not in st.session_state:
    st.session_state["flagged_cache"] = flagged_df.copy() if not flagged_df.empty else pd.DataFrame()

# keep caches updated
if vendor_agg is not None and not vendor_agg.empty:
    st.session_state["vendor_agg_cache"] = vendor_agg.copy()
if flagged_df is not None and not flagged_df.empty:
    st.session_state["flagged_cache"] = flagged_df.copy()

chat_col, input_col = st.columns([3,1])
with chat_col:
    if st.session_state["chat_history"]:
        for msg in st.session_state["chat_history"]:
            role = msg.get("role", "user")
            content = msg.get("message", "")
            if role == "user":
                with st.chat_message("user"):
                    st.markdown(content)
            else:
                with st.chat_message("assistant"):
                    st.markdown(content)
    else:
        st.info("Belum ada chat. Mulai tanya sesuatu tentang potensi fraud atau outlier di data.")

with input_col:
    user_input = st.chat_input("Tanyakan sesuatu tentang potensi penipuan / transaksi mencurigakan.")
    if user_input:
        st.session_state["chat_history"].append({"role": "user", "message": user_input})
        response_text = ""
        # Prefer GROQ if available
        if GROQ_API_KEY and has_groq and groq_client is not None:
            try:
                conv = ["Anda adalah analis risiko/penipuan. Jawab singkat dalam bahasa Indonesia."]
                for h in st.session_state["chat_history"][-6:]:
                    conv.append(f"{h['role'].upper()}: {h['message']}")
                conv.append("\nRingkasan TotalAmount per Vendor:")
                if not st.session_state["vendor_agg_cache"].empty:
                    total_all = st.session_state["vendor_agg_cache"]["TotalAmount"].sum()
                    for _, r in st.session_state["vendor_agg_cache"].iterrows():
                        pct = (r["TotalAmount"] / total_all * 100) if total_all else 0
                        conv.append(f"- {r['Vendor']}: {r['TotalAmount']:.2f} ({pct:.1f}%)")
                else:
                    conv.append("- (tidak ada ringkasan)")
                conv.append("\nTransaksi ter-flag (5 teratas):")
                if not st.session_state["flagged_cache"].empty:
                    sample = st.session_state["flagged_cache"].head(5)
                    for _, r in sample.iterrows():
                        tid = r.get("TransactionID", "N/A")
                        amt = r.get("Amount", "N/A")
                        conv.append(f"- {tid}: {amt}")
                else:
                    conv.append("- (tidak ada transaksi ter-flag)")
                prompt_for_chat = "\n".join(conv) + f"\n\nPertanyaan: {user_input}\nJawab singkat, to the point."
                resp = groq_client.generate(
                    model="llama-3.3-70b-versatile",
                    prompt=prompt_for_chat,
                    max_tokens=500
                )
                if isinstance(resp, dict) and "text" in resp:
                    response_text = resp["text"].strip()
                else:
                    response_text = getattr(resp, "text", str(resp)).strip()
            except Exception as e:
                response_text = f"⚠️ Gagal memanggil AI: {e}"
        else:
            # fallback
            context = {"vendor_agg": st.session_state.get("vendor_agg_cache", pd.DataFrame()),
                       "flagged_df": st.session_state.get("flagged_cache", pd.DataFrame())}
            response_text = simple_ai_reply_chat(user_input, context)

        st.session_state["chat_history"].append({"role": "assistant", "message": response_text})
        # Streamlit will rerun to display the new messages

# ---------- Footer notes / Error handling ----------
st.markdown("---")
st.markdown("**Catatan & Error handling:**")
st.markdown("- Jika `GROQ_API_KEY` tidak ditemukan, AI Commentary non-aktif dan chat fallback menggunakan heuristik lokal. ⚠️")
st.markdown("- Pastikan file yang diunggah memiliki kolom `Amount` dan `Vendor` untuk analisis utama. Kolom `EmployeeID`, `Account`, `Time` sangat membantu investigasi.")
st.markdown("- Deteksi outlier yang disediakan adalah heuristik awal (IQR, z-score, threshold). Untuk investigasi forensik, gunakan metode statistik/ML lanjutan dan cross-check metadata transaksi.")
st.markdown("- Semua pemanggilan eksternal dibungkus try/except sehingga kegagalan API tidak merusak UI.")
