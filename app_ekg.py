# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import streamlit as st
import plotly.graph_objects as go
from PyEMD import EMD

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

# Fix dla widoczności napisów i estetyki
st.markdown(f"""
    <style>
    /* Główny tekst aplikacji na biało */
    .stApp {{ color: {bialy}; }}
    
    /* Nagłówki */
    h1, h2, h3, [data-testid="stHeader"] {{ color: {bialy} !important; }}
    
    /* Etykiety nad polami input (naprawa Twojego "zniknięcia") */
    [data-testid="stWidgetLabel"] p, label {{
        color: {bialy} !important;
        font-weight: 500 !important;
    }}
    
    /* Tekst w tabelach i ogólny */
    p, .stText {{ color: {bialo_szary}; }}

    /* Ramka nagłówka */
    .moja-ramka {{ 
        border-radius:10px; 
        padding:20px; 
        background-color:{czerwony};
        text-align:center; 
        margin-bottom: 20px;
    }}
    .moja-ramka h4 {{ color:{czarny} !important; margin:0; font-weight: bold; }}
    </style>
""", unsafe_allow_html=True)

# ── FUNKCJE POMOCNICZE ───────────────────────────────────────────────────────

def downsample(x, y, max_points=2000):
    """Ogranicza liczbę punktów na wykresie dla płynności działania."""
    n = len(x)
    if n <= max_points:
        return x, y
    idx = np.round(np.linspace(0, n - 1, max_points)).astype(int)
    return x[idx], y[idx]

@st.cache_data
def load_my_data(file_path):
    """Wczytywanie danych z obsługą błędów."""
    try:
        data = pd.read_csv(file_path, sep='\t', decimal=',', skiprows=6,
            header=None, engine='python', encoding='cp1250', on_bad_lines='skip')
        data = data.iloc[:, :2].copy()
        data.columns = ['czas', 'ecg']
        data['czas'] = data['czas'].astype(str).str.replace(',', '.', regex=False)
        data['ecg']  = data['ecg'].astype(str).str.replace(',', '.', regex=False)
        data = data.apply(pd.to_numeric, errors='coerce')
        return data.dropna().reset_index(drop=True)
    except Exception as e:
        st.error(f"Nie udało się wczytać pliku {file_path}: {e}")
        return pd.DataFrame(columns=['czas', 'ecg'])

@st.cache_data
def cached_emd(ecg_bytes):
    """Obliczenia EMD schowane w cache, by nie liczyć tego co zmianę suwaka."""
    ecg_array = np.frombuffer(ecg_bytes, dtype=np.float64)
    emd_obj = EMD()
    imf = emd_obj(ecg_array).T 
    n_imf = imf.shape[1]
    
    # Automatyczny wybór trendu (ostatnie składowe)
    if n_imf >= 8:
        baseline = imf[:, 6] + imf[:, 7]
    elif n_imf >= 2:
        baseline = imf[:, -1] + imf[:, -2]
    else:
        baseline = imf[:, -1]
        
    return imf, baseline, ecg_array - baseline

def detect_r_peaks(df_in, distance_ms=500, height=None, sampling_rate=1000):
    signal = df_in["ecg"].values
    distance_samples = int(distance_ms * sampling_rate / 1000)
    peaks, _ = find_peaks(signal, distance=distance_samples, height=height)
    return peaks

def compute_rr(czas, peaks):
    r_times = czas.iloc[peaks].values
    rr_s = np.diff(r_times)
    return rr_s, rr_s * 1000, r_times[1:], r_times

def compute_hrv_metrics(rr_ms):
    if len(rr_ms) < 2:
        return {"mean_rr": 0, "sdnn": 0, "rmssd": 0, "pnn50": 0, "min_rr": 0, "max_rr": 0}
    diff_rr = np.diff(rr_ms)
    return {
        "mean_rr": np.mean(rr_ms),
        "sdnn": np.std(rr_ms, ddof=1),
        "rmssd": np.sqrt(np.mean(diff_rr**2)),
        "pnn50": 100 * np.sum(np.abs(diff_rr) > 50) / len(diff_rr),
        "min_rr": np.min(rr_ms),
        "max_rr": np.max(rr_ms),
    }

def export_ecg_txt(czas, ecg_clean):
    out_df = pd.DataFrame({"czas[s]": np.round(czas, 6), "ECG": np.round(ecg_clean, 6)})
    return out_df.to_csv(sep="\t", index=False).encode("utf-8")

# ── WCZYTYWANIE DANYCH ───────────────────────────────────────────────────────
df_spoczynek = load_my_data("Spoczynek.txt")
df_wysilek   = load_my_data("Wysilek.txt")
if not df_wysilek.empty:
    df_wysilek["ecg"] = df_wysilek["ecg"] / 500

# ── NAGŁÓWEK ─────────────────────────────────────────────────────────────────
st.markdown(f"""
    <div class="moja-ramka">
        <h4>Analiza HRV sygnału EKG</h4>
        <p style="color:{bialy};">laboratorium fizyki medycznej</p>
    </div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SEKCJA 1 — Podgląd sygnałów
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f'<p style="font-size:18px;font-weight:bold;color:{niebieski};">Podgląd sygnałów</p>', unsafe_allow_html=True)

col_top_left, col_top_mid, col_top_right = st.columns([1.6, 1.6, 4.8])

with col_top_left:
    start_sp = st.number_input("Start Spoczynek [s]", value=0.0, step=0.1, key="s1")
    end_sp = st.number_input("Koniec Spoczynek [s]", value=10.0, step=0.1, key="e1")
    df_sp_view = df_spoczynek[(df_spoczynek["czas"] >= start_sp) & (df_spoczynek["czas"] <= end_sp)]
    st.dataframe(df_sp_view.head(50), height=250, use_container_width=True)

with col_top_mid:
    start_wy = st.number_input("Start Wysiłek [s]", value=0.0, step=0.1, key="s2")
    end_wy = st.number_input("Koniec Wysiłek [s]", value=10.0, step=0.1, key="e2")
    df_wy_view = df_wysilek[(df_wysilek["czas"] >= start_wy) & (df_wysilek["czas"] <= end_wy)]
    st.dataframe(df_wy_view.head(50), height=250, use_container_width=True)

with col_top_right:
    fig_compare = go.Figure()
    # Spoczynek
    sx, sy = downsample(df_sp_view["czas"].values, df_sp_view["ecg"].values)
    fig_compare.add_trace(go.Scatter(x=sx, y=sy, mode="lines", name="Spoczynek", line=dict(color=niebieski, width=1.5)))
    # Wysiłek
    wx, wy = downsample(df_wy_view["czas"].values, df_wy_view["ecg"].values)
    fig_compare.add_trace(go.Scatter(x=wx, y=wy, mode="lines", name="Wysiłek", line=dict(color=lekki_czerwony, width=1.5)))

    fig_compare.update_layout(
        height=320, margin=dict(l=40, r=10, t=10, b=40),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=bialy), # Wymuszenie białych osi
        xaxis=dict(gridcolor="#333", title="Czas [s]"),
        yaxis=dict(gridcolor="#333", title="Amplituda [mV]"),
        legend=dict(orientation="h", y=1.1)
    )
    st.plotly_chart(fig_compare, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# SEKCJA 2 — Analiza RR
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f'<p style="font-size:18px;font-weight:bold;color:{niebieski};">Analiza RR</p>', unsafe_allow_html=True)
st.markdown(f'<hr style="border:none;height:2px;background-color:{niebieski};margin-bottom:20px;" />', unsafe_allow_html=True)

col_rr_left, col_rr_right = st.columns([1.7, 4.3])

with col_rr_left:
    sygnal = st.selectbox("Wybierz sygnał", ["Spoczynek", "Wysiłek"])
    df_active = df_spoczynek if sygnal == "Spoczynek" else df_wysilek
    
    start_rr = st.number_input("Start RR [s]", value=0.0, step=1.0)
    end_rr = st.number_input("Koniec RR [s]", value=30.0, step=1.0)
    prog_r = st.number_input("Próg piku R", value=0.4, step=0.05)
    dystans_r = st.number_input("Min. odstęp [ms]", value=500, step=50)
    
    df_rr = df_active[(df_active["czas"] >= start_rr) & (df_active["czas"] <= end_rr)].copy()
    peaks = detect_r_peaks(df_rr, distance_ms=dystans_r, height=prog_r)
    rr_s, rr_ms, rr_time, r_times = compute_rr(df_rr["czas"], peaks)
    metrics = compute_hrv_metrics(rr_ms)

with col_rr_right:
    fig_rr = go.Figure()
    rx, ry = downsample(df_rr["czas"].values, df_rr["ecg"].values)
    fig_rr.add_trace(go.Scatter(x=rx, y=ry, mode="lines", name="EKG", line=dict(color=bialo_szary, width=1)))
    if len(peaks) > 0:
        fig_rr.add_trace(go.Scatter(x=df_rr["czas"].iloc[peaks], y=df_rr["ecg"].iloc[peaks], 
                                     mode="markers", name="Piki R", marker=dict(color=niebieski, size=8)))
    
    fig_rr.update_layout(height=300, margin=dict(l=40, r=10, t=10, b=40),
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font=dict(color=bialy), xaxis=dict(gridcolor="#333"), yaxis=dict(gridcolor="#333"))
    st.plotly_chart(fig_rr, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# SEKCJA 3 — Metryki i Histogram
# ══════════════════════════════════════════════════════════════════════════════
col_met, col_hist = st.columns([2, 3])

with col_met:
    m1, m2 = st.columns(2)
    m1.metric("Średnie RR", f"{metrics['mean_rr']:.1f} ms")
    m2.metric("SDNN", f"{metrics['sdnn']:.1f} ms")
    m3, m4 = st.columns(2)
    m3.metric("RMSSD", f"{metrics['rmssd']:.1f} ms")
    m4.metric("pNN50", f"{metrics['pnn50']:.1f} %")

with col_hist:
    fig_h = go.Figure(go.Histogram(x=rr_ms, marker_color=lekki_czerwony, nbinsx=15))
    fig_h.update_layout(height=250, margin=dict(l=40, r=10, t=10, b=40),
                         paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                         font=dict(color=bialy), xaxis=dict(title="RR [ms]"))
    st.plotly_chart(fig_h, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# SEKCJA 4 — EMD
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f'<p style="font-size:18px;font-weight:bold;color:{niebieski};">Usuwanie trendu (EMD)</p>', unsafe_allow_html=True)
st.markdown(f'<hr style="border:none;height:2px;background-color:{niebieski};margin-bottom:20px;" />', unsafe_allow_html=True)

# Krótki zakres dla EMD żeby nie zamuliło (EMD jest ciężkie obliczeniowo)
df_emd_slice = df_active[(df_active["czas"] >= start_rr) & (df_active["czas"] <= start_rr + 15)].copy()

if len(df_emd_slice) > 100:
    with st.spinner("Liczenie EMD (15s sygnału)..."):
        imf, base, clean = cached_emd(df_emd_slice["ecg"].values.astype(np.float64).tobytes())
    
    fig_emd = go.Figure()
    ex, ey = downsample(df_emd_slice["czas"].values, df_emd_slice["ecg"].values)
    _, eb = downsample(df_emd_slice["czas"].values, base)
    _, ec = downsample(df_emd_slice["czas"].values, clean)
    
    fig_emd.add_trace(go.Scatter(x=ex, y=ey, name="Oryginał", line=dict(color=mocny_szary, width=1)))
    fig_emd.add_trace(go.Scatter(x=ex, y=eb, name="Trend", line=dict(color=lekki_czerwony, width=2)))
    fig_emd.add_trace(go.Scatter(x=ex, y=ec, name="Oczyszczony", line=dict(color=niebieski, width=1.2)))
    
    fig_emd.update_layout(height=350, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font=dict(color=bialy), legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig_emd, use_container_width=True)
    
    # Pobieranie
    csv_clean = export_ecg_txt(df_emd_slice["czas"].values, clean)
    st.download_button("Pobierz oczyszczony sygnał (.txt)", data=csv_clean, file_name="EKG_clean.txt")
else:
    st.info("Wybierz zakres czasu w sekcji RR, aby zobaczyć EMD.")
