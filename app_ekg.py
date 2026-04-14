# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import streamlit as st
import plotly.graph_objects as go
from PyEMD import EMD
import requests
import io

# Konfiguracja strony
st.set_page_config(layout="wide", page_title="Analiza HRV EKG", page_icon="❤️")

# ── KOLORY I STYLIZACJA ──────────────────────────────────────────────────────
bialy          = "#ffffff"
bialo_szary    = "#aeaeae"
niebieski      = "#0092ff"
czerwony       = "#ff1100"
lekki_szary    = "#363636"
lekki_czerwony = "#e74c3c"
mocny_szary    = "#999999"
czarny         = "#000000"

st.markdown(f"""
    <style>
    .stApp {{ color: {bialy}; }}
    h1, h2, h3, [data-testid="stHeader"] {{ color: {bialy} !important; }}
    [data-testid="stWidgetLabel"] p, label {{ color: {bialy} !important; font-weight: 500 !important; }}
    p, .stText {{ color: {bialo_szary}; }}
    .moja-ramka {{ 
        border-radius:10px; padding:20px; background-color:{czerwony};
        text-align:center; margin-bottom: 20px;
    }}
    .moja-ramka h4 {{ color:{czarny} !important; margin:0; font-weight: bold; }}
    </style>
""", unsafe_allow_html=True)

# ── FUNKCJE POMOCNICZE ───────────────────────────────────────────────────────

def downsample(x, y, max_points=2000):
    n = len(x)
    if n <= max_points: return x, y
    idx = np.round(np.linspace(0, n - 1, max_points)).astype(int)
    return x[idx], y[idx]

@st.cache_data
def load_data_from_drive(file_id):
    """Pobiera plik z Google Drive korzystając z ID."""
    url = f'https://drive.google.com/uc?export=download&id={file_id}'
    try:
        response = requests.get(url)
        response.raise_for_status()
        # Dekodowanie zawartości
        raw_data = response.content.decode('cp1250')
        
        data = pd.read_csv(io.StringIO(raw_data), sep='\t', decimal=',', skiprows=6,
                           header=None, engine='python', on_bad_lines='skip')
        data = data.iloc[:, :2].copy()
        data.columns = ['czas', 'ecg']
        data['czas'] = data['czas'].astype(str).str.replace(',', '.', regex=False)
        data['ecg']  = data['ecg'].astype(str).str.replace(',', '.', regex=False)
        data = data.apply(pd.to_numeric, errors='coerce')
        return data.dropna().reset_index(drop=True)
    except Exception as e:
        st.error(f"Błąd przy pobieraniu pliku (ID: {file_id}): {e}")
        return pd.DataFrame(columns=['czas', 'ecg'])

@st.cache_data
def cached_emd(ecg_bytes):
    ecg_array = np.frombuffer(ecg_bytes, dtype=np.float64)
    emd_obj = EMD()
    imf = emd_obj(ecg_array).T 
    n_imf = imf.shape[1]
    if n_imf >= 8: baseline = imf[:, 6] + imf[:, 7]
    elif n_imf >= 2: baseline = imf[:, -1] + imf[:, -2]
    else: baseline = imf[:, -1]
    return imf, baseline, ecg_array - baseline

def detect_r_peaks(df_in, distance_ms=500, height=None):
    # Zakładamy fs = 1000Hz (odstęp 500ms = 500 próbek)
    peaks, _ = find_peaks(df_in["ecg"].values, distance=500, height=height)
    return peaks

def compute_rr(czas, peaks):
    r_times = czas.iloc[peaks].values
    rr_s = np.diff(r_times)
    return rr_s, rr_s * 1000, r_times[1:]

def compute_hrv_metrics(rr_ms):
    if len(rr_ms) < 2: return {"mean_rr": 0, "sdnn": 0, "rmssd": 0, "pnn50": 0, "min_rr": 0, "max_rr": 0}
    diff_rr = np.diff(rr_ms)
    return {
        "mean_rr": np.mean(rr_ms), "sdnn": np.std(rr_ms, ddof=1),
        "rmssd": np.sqrt(np.mean(diff_rr**2)), "pnn50": 100 * np.sum(np.abs(diff_rr) > 50) / len(diff_rr),
        "min_rr": np.min(rr_ms), "max_rr": np.max(rr_ms),
    }

def export_ecg_txt(czas, ecg_clean):
    out_df = pd.DataFrame({"czas[s]": np.round(czas, 6), "ECG": np.round(ecg_clean, 6)})
    return out_df.to_csv(sep="\t", index=False).encode("utf-8")

# ── KONFIGURACJA ID PLIKÓW ───────────────────────────────────────────────────
ID_SPOCZYNEK = "1B_2MfGY_EPqY1dZmbLzHX6j8eod98huA"
ID_WYSILEK   = "1OJrkeyTIkGPHiTGMYsYIyWGVdo-Tov4t"

# ── DATA LOADING ─────────────────────────────────────────────────────────────
df_spoczynek = load_data_from_drive(ID_SPOCZYNEK)
df_wysilek   = load_data_from_drive(ID_WYSILEK)

if not df_wysilek.empty: 
    df_wysilek["ecg"] = df_wysilek["ecg"] / 500

# ── HEADER ───────────────────────────────────────────────────────────────────
st.markdown(f'<div class="moja-ramka"><h4>Analiza HRV sygnału EKG</h4><p style="color:{bialy};">laboratorium fizyki medycznej</p></div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Signal preview
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f'<p style="font-size:18px;font-weight:bold;color:{niebieski};">Podgląd sygnałów (Google Drive)</p>', unsafe_allow_html=True)
col_top_left, col_top_mid, col_top_right = st.columns([1.6, 1.6, 4.8])

with col_top_left:
    start_sp = st.number_input("Początek Spoczynek [s]", value=0.0, step=1.0)
    end_sp = st.number_input("Koniec Spoczynek [s]", value=10.0, step=1.0)
    df_sp_view = df_spoczynek[(df_spoczynek["czas"] >= start_sp) & (df_spoczynek["czas"] <= end_sp)]
    st.dataframe(df_sp_view.head(50), height=250, use_container_width=True)

with col_top_mid:
    start_wy = st.number_input("Początek Wysiłek [s]", value=0.0, step=1.0)
    end_wy = st.number_input("Koniec Wysiłek [s]", value=10.0, step=1.0)
    df_wy_view = df_wysilek[(df_wysilek["czas"] >= start_wy) & (df_wysilek["czas"] <= end_wy)]
    st.dataframe(df_wy_view.head(50), height=250, use_container_width=True)

with col_top_right:
    fig_comp = go.Figure()
    if not df_sp_view.empty:
        sx, sy = downsample(df_sp_view["czas"].values, df_sp_view["ecg"].values)
        fig_comp.add_trace(go.Scatter(x=sx, y=sy, mode="lines", name="Spoczynek", line=dict(color=niebieski)))
    if not df_wy_view.empty:
        wx, wy = downsample(df_wy_view["czas"].values, df_wy_view["ecg"].values)
        fig_comp.add_trace(go.Scatter(x=wx, y=wy, mode="lines", name="Wysiłek", line=dict(color=lekki_czerwony)))
    fig_comp.update_layout(height=320, margin=dict(l=40, r=10, t=10, b=40), paper_bgcolor="rgba(0,0,0,0)", 
                           plot_bgcolor="rgba(0,0,0,0)", font=dict(color=bialy), legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig_comp, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — RR analysis
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f'<p style="font-size:18px;font-weight:bold;color:{niebieski};">Analiza RR</p>', unsafe_allow_html=True)
col_rr_l, col_rr_r = st.columns([1.7, 4.3])

with col_rr_l:
    sygnal = st.selectbox("Sygnał", ["Spoczynek", "Wysiłek"])
    df_active = df_spoczynek if sygnal == "Spoczynek" else df_wysilek
    start_rr = st.number_input("Start RR [s]", value=0.0)
    end_rr = st.number_input("Koniec RR [s]", value=30.0)
    prog_r = st.number_input("Próg R", value=0.5)
    df_rr = df_active[(df_active["czas"] >= start_rr) & (df_active["czas"] <= end_rr)].copy()
    peaks = detect_r_peaks(df_rr, height=prog_r)
    rr_s, rr_ms, rr_time = compute_rr(df_rr["czas"], peaks)
    metrics = compute_hrv_metrics(rr_ms)

with col_rr_r:
    fig_rr = go.Figure()
    rx, ry = downsample(df_rr["czas"].values, df_rr["ecg"].values)
    fig_rr.add_trace(go.Scatter(x=rx, y=ry, mode="lines", name="EKG", line=dict(color=bialo_szary, width=1)))
    fig_rr.add_trace(go.Scatter(x=df_rr["czas"].iloc[peaks], y=df_rr["ecg"].iloc[peaks], mode="markers", name="Piki R", marker=dict(color=niebieski, size=8)))
    fig_rr.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color=bialy))
    st.plotly_chart(fig_rr, use_container_width=True)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Średnie RR", f"{metrics['mean_rr']:.1f} ms")
m2.metric("SDNN", f"{metrics['sdnn']:.1f} ms")
m3.metric("RMSSD", f"{metrics['rmssd']:.1f} ms")
m4.metric("pNN50", f"{metrics['pnn50']:.1f} %")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — EMD (ROZŁOŻENIE NA SKŁADOWE)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f'<p style="font-size:18px;font-weight:bold;color:{niebieski};">Dekompozycja EMD (Częstotliwości)</p>', unsafe_allow_html=True)
col_emd_l, col_emd_r = st.columns([1.7, 4.3])

df_emd = df_active[(df_active["czas"] >= start_rr) & (df_active["czas"] <= start_rr + 15)].copy()

if len(df_emd) > 100:
    with st.spinner("Przetwarzanie EMD..."):
        imfs, trend, clean = cached_emd(df_emd["ecg"].values.astype(np.float64).tobytes())
    
    with col_emd_l:
        st.write("Wybierz IMF do rekonstrukcji:")
        imf_ops = [f"IMF {i+1}" for i in range(imfs.shape[1])]
        selected = st.multiselect("Składowe", imf_ops, default=imf_ops[:3])
        st.download_button("Pobierz czyste EKG", data=export_ecg_txt(df_emd["czas"].values, clean), file_name="EKG_EMD.txt")

    with col_emd_r:
        fig_emd_main = go.Figure()
        ex, ey = downsample(df_emd["czas"].values, df_emd["ecg"].values)
        _, et = downsample(df_emd["czas"].values, trend)
        _, ec = downsample(df_emd["czas"].values, clean)
        fig_emd_main.add_trace(go.Scatter(x=ex, y=ey, name="Surowy", line=dict(color=mocny_szary, width=1)))
        fig_emd_main.add_trace(go.Scatter(x=ex, y=et, name="Trend (Niska f)", line=dict(color=lekki_czerwony, width=2)))
        fig_emd_main.add_trace(go.Scatter(x=ex, y=ec, name="Oczyszczony", line=dict(color=niebieski, width=1)))
        fig_emd_main.update_layout(height=350, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color=bialy))
        st.plotly_chart(fig_emd_main, use_container_width=True)

    st.write("### Poszczególne składowe sygnału (od wysokich do niskich częstotliwości):")
    n_show = min(imfs.shape[1], 8)
    cols_imf = st.columns(2)
    imf_colors = ["#0092ff", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c", "#d35400", "#34495e"]
    
    for i in range(n_show):
        target_col = cols_imf[0] if i % 2 == 0 else cols_imf[1]
        with target_col:
            ix, iy = downsample(df_emd["czas"].values, imfs[:, i])
            f_imf = go.Figure(go.Scatter(x=ix, y=iy, line=dict(color=imf_colors[i % 8], width=1.5)))
            f_imf.update_layout(height=180, title=f"IMF {i+1}", margin=dict(l=10, r=10, t=30, b=10),
                                 paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(30,30,30,0.2)",
                                 font=dict(color=bialy, size=10))
            st.plotly_chart(f_imf, use_container_width=True)
else:
    st.info("Wybierz zakres czasu w sekcji RR, aby uruchomić EMD.")
