# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import streamlit as st
import plotly.graph_objects as go
from PyEMD import EMD

st.set_page_config(layout="wide", page_title="Analiza HRV EKG", page_icon="❤️")

bialy          = "#ffffff"
bialo_szary    = "#aeaeae"
niebieski      = "#0092ff"
czerwony       = "#ff1100"
lekki_szary    = "#363636"
lekki_czerwony = "#e74c3c"
mocny_szary    = "#999999"
czarny         = "#000000"
charcoal       = "#36454F"

st.markdown(f"""
    <style>
    .stApp {{ color: {czarny}; }}
    h1, h2, h3, [data-testid="stHeader"] {{ color: {bialy} !important; }}
    [data-testid="stMetricValue"] {{ font-size: 18px !important; color: {bialy} !important; }}
    [data-testid="stMetricLabel"] p {{ color: {mocny_szary} !important; }}
    </style>
""", unsafe_allow_html=True)

# ── IMPROVEMENT 1: downsample helper ─────────────────────────────────────────
def downsample(x, y, max_points=3000):
    """Keep at most max_points evenly-spaced samples for display."""
    n = len(x)
    if n <= max_points:
        return x, y
    idx = np.round(np.linspace(0, n - 1, max_points)).astype(int)
    return x[idx], y[idx]

# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_my_data(file):
    data = pd.read_csv(file, sep='\t', decimal=',', skiprows=6,
        header=None, engine='python', encoding='cp1250', on_bad_lines='skip')
    data = data.iloc[:, :2].copy()
    data.columns = ['czas', 'ecg']
    data['czas'] = data['czas'].astype(str).str.replace(',', '.', regex=False)
    data['ecg']  = data['ecg'].astype(str).str.replace(',', '.', regex=False)
    data = data.apply(pd.to_numeric, errors='coerce')
    data = data.dropna().reset_index(drop=True)
    return data

# ── IMPROVEMENT 2: cache the expensive EMD call ───────────────────────────────
@st.cache_data
def cached_emd(ecg_bytes):
    ecg_array = np.frombuffer(ecg_bytes, dtype=np.float64)
    emd_obj = EMD()
    imf = emd_obj(ecg_array).T  # PyEMD returns (n_imfs, n_samples), transpose to (n_samples, n_imfs)
    n_imf = imf.shape[1]
    if n_imf >= 9:
        baseline = imf[:, 7] + imf[:, 8]
    elif n_imf >= 8:
        baseline = imf[:, 6] + imf[:, 7]
    elif n_imf >= 7:
        baseline = imf[:, 5] + imf[:, 6]
    elif n_imf >= 2:
        baseline = imf[:, -1] + imf[:, -2]
    else:
        baseline = imf[:, -1]
    ecg_detrended = ecg_array - baseline
    return imf, baseline, ecg_detrended

def detect_r_peaks(df_in, distance_ms=500, height=None, sampling_rate=1000):
    signal = df_in["ecg"].values
    distance_samples = int(distance_ms * sampling_rate / 1000)
    peaks, properties = find_peaks(signal, distance=distance_samples, height=height)
    return peaks, properties

def compute_rr(czas, peaks):
    r_times = czas.iloc[peaks].values
    rr_s  = np.diff(r_times)
    rr_ms = rr_s * 1000
    rr_time = r_times[1:]
    return rr_s, rr_ms, rr_time, r_times

def compute_hrv_metrics(rr_ms):
    if len(rr_ms) == 0:
        return {"mean_rr": 0, "sdnn": 0, "rmssd": 0, "pnn50": 0, "min_rr": 0, "max_rr": 0}
    diff_rr = np.diff(rr_ms)
    return {
        "mean_rr": np.mean(rr_ms),
        "sdnn":    np.std(rr_ms, ddof=1) if len(rr_ms) > 1 else 0,
        "rmssd":   np.sqrt(np.mean(diff_rr**2)) if len(diff_rr) > 0 else 0,
        "pnn50":   100 * np.sum(np.abs(diff_rr) > 50) / len(diff_rr) if len(diff_rr) > 0 else 0,
        "min_rr":  np.min(rr_ms),
        "max_rr":  np.max(rr_ms),
    }

def export_ecg_txt(czas, ecg_clean):
    out_df = pd.DataFrame({"czas[s]": np.round(czas, 6), "ECG": np.round(ecg_clean, 6)})
    return out_df.to_csv(sep="\t", index=False).encode("utf-8")

# ── Load & scale data ─────────────────────────────────────────────────────────
df_spoczynek = load_my_data("Spoczynek.txt")
df_wysilek   = load_my_data("Wysilek.txt")
df_wysilek["ecg"] = df_wysilek["ecg"] / 500

# ── Title ─────────────────────────────────────────────────────────────────────
st.markdown(f"""
    <style>
    .moja-ramka {{ border-radius:10px; padding:20px; background-color:{czerwony};
        text-align:center; height:120px; }}
    .moja-ramka h4 {{ color:{czarny}; margin:0; }}
    </style>
    <div class="moja-ramka">
        <h4>Analiza HRV sygnału EKG</h4>
        <p style="color:{bialy};">laboratorium fizyki medycznej</p>
    </div>
""", unsafe_allow_html=True)
st.markdown('<hr style="margin-top:10px;height:5px;border:none;background-color:#444444;" />',
    unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Signal preview
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<p style="font-size:18px;font-weight:bold;color:#0092ff;">Podgląd sygnałów</p>',
    unsafe_allow_html=True)

col_top_left, col_top_mid, col_top_right = st.columns([1.6, 1.6, 4.8])

with col_top_left:
    start_sp = st.number_input("Początek zakresu Spoczynek [s]",
        min_value=float(df_spoczynek["czas"].min()), max_value=float(df_spoczynek["czas"].max()),
        value=float(df_spoczynek["czas"].min()), step=0.1)
    end_sp = st.number_input("Koniec zakresu Spoczynek [s]",
        min_value=float(df_spoczynek["czas"].min()), max_value=float(df_spoczynek["czas"].max()),
        value=min(float(df_spoczynek["czas"].min()) + 10, float(df_spoczynek["czas"].max())), step=0.1)
    if end_sp < start_sp:
        st.warning("Koniec zakresu Spoczynek musi być większy lub równy początkowi.")
    df_sp_view = df_spoczynek[(df_spoczynek["czas"] >= start_sp) & (df_spoczynek["czas"] <= end_sp)].copy()
    n_sp = st.number_input("Liczba wierszy tabeli Spoczynek", min_value=1,
        max_value=max(1, len(df_sp_view)), value=min(50, max(1, len(df_sp_view))), step=1)
    st.dataframe(df_sp_view.head(int(n_sp)), height=260, use_container_width=True)

with col_top_mid:
    start_wy = st.number_input("Początek zakresu Wysiłek [s]",
        min_value=float(df_wysilek["czas"].min()), max_value=float(df_wysilek["czas"].max()),
        value=float(df_wysilek["czas"].min()), step=0.1)
    end_wy = st.number_input("Koniec zakresu Wysiłek [s]",
        min_value=float(df_wysilek["czas"].min()), max_value=float(df_wysilek["czas"].max()),
        value=min(float(df_wysilek["czas"].min()) + 10, float(df_wysilek["czas"].max())), step=0.1)
    if end_wy < start_wy:
        st.warning("Koniec zakresu Wysiłek musi być większy lub równy początkowi.")
    df_wy_view = df_wysilek[(df_wysilek["czas"] >= start_wy) & (df_wysilek["czas"] <= end_wy)].copy()
    n_wy = st.number_input("Liczba wierszy tabeli Wysiłek", min_value=1,
        max_value=max(1, len(df_wy_view)), value=min(50, max(1, len(df_wy_view))), step=1)
    st.dataframe(df_wy_view.head(int(n_wy)), height=260, use_container_width=True)

# IMPROVEMENT 3: right-side chart now shows only the selected ranges, not full signals
with col_top_right:
    fig_compare = go.Figure()

    # Spoczynek — only selected range, downsampled
    sp_x, sp_y = downsample(df_sp_view["czas"].values, df_sp_view["ecg"].values)
    fig_compare.add_trace(go.Scattergl(x=sp_x, y=sp_y,
        mode="lines", name="Spoczynek", line=dict(color=niebieski, width=1.5)))

    # Wysilek — only selected range, downsampled
    wy_x, wy_y = downsample(df_wy_view["czas"].values, df_wy_view["ecg"].values)
    fig_compare.add_trace(go.Scattergl(x=wy_x, y=wy_y,
        mode="lines", name="Wysiłek", line=dict(color=lekki_czerwony, width=1.5)))

    fig_compare.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis_title="Czas [s]", yaxis_title="Amplituda [mV]")
    with st.container(border=True):
        st.plotly_chart(fig_compare, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — RR analysis
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<p style="font-size:18px;font-weight:bold;color:#0092ff;">Analiza RR</p>',
    unsafe_allow_html=True)
st.markdown(f'<hr style="margin-top:10px;height:5px;border:none;background-color:{niebieski};" />',
    unsafe_allow_html=True)

col_rr_left, col_rr_right = st.columns([1.7, 4.3])

with col_rr_left:
    sygnal    = st.selectbox("Wybierz sygnał do RR", ["Spoczynek", "Wysiłek"])
    df_active = df_spoczynek.copy() if sygnal == "Spoczynek" else df_wysilek.copy()
    start_rr  = st.number_input("Początek zakresu RR [s]",
        min_value=float(df_active["czas"].min()), max_value=float(df_active["czas"].max()),
        value=float(df_active["czas"].min()), step=0.1)
    end_rr = st.number_input("Koniec zakresu RR [s]",
        min_value=float(df_active["czas"].min()), max_value=float(df_active["czas"].max()),
        value=min(float(df_active["czas"].min()) + 20, float(df_active["czas"].max())), step=0.1)
    prog_r = st.number_input("Próg dla pików R",
        min_value=float(df_active["ecg"].min()), max_value=float(df_active["ecg"].max()),
        value=float(df_active["ecg"].max()) * 0.6, step=0.01)
    dystans_r    = st.number_input("Minimalny odstęp między R [ms]", min_value=200, max_value=1200, value=500, step=10)
    liczba_binow = st.number_input("Liczba binów histogramu RR", min_value=5, max_value=50, value=20, step=1)
    if end_rr <= start_rr:
        st.warning("Koniec zakresu RR musi być większy niż początek.")
    df_rr = df_active[(df_active["czas"] >= start_rr) & (df_active["czas"] <= end_rr)].copy()
    if len(df_rr) > 0 and end_rr > start_rr:
        peaks, _ = detect_r_peaks(df_rr, sampling_rate=1000, distance_ms=int(dystans_r), height=prog_r)
    else:
        peaks = np.array([], dtype=int)
    rr_s, rr_ms, rr_time, r_times = compute_rr(df_rr["czas"], peaks)
    metrics = compute_hrv_metrics(rr_ms)

with col_rr_right:
    fig_rr_signal = go.Figure()
    # Downsample ECG signal for display — R peaks are always shown in full
    rr_x, rr_y = downsample(df_rr["czas"].values, df_rr["ecg"].values)
    fig_rr_signal.add_trace(go.Scattergl(x=rr_x, y=rr_y,
        mode="lines", name="Sygnał EKG", line=dict(color=bialo_szary, width=1.5)))
    if len(peaks) > 0:
        fig_rr_signal.add_trace(go.Scattergl(
            x=df_rr["czas"].iloc[peaks], y=df_rr["ecg"].iloc[peaks],
            mode="markers", name="Piki R", marker=dict(color=niebieski, size=7)))
    fig_rr_signal.update_layout(height=260, margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis_title="Czas [s]", yaxis_title="Amplituda [mV]")
    with st.container(border=True):
        st.plotly_chart(fig_rr_signal, use_container_width=True)

col_rr_tab, col_rr_tacho = st.columns([2.5, 2.5])
with col_rr_tab:
    rr_df = pd.DataFrame({"#": np.arange(1, len(rr_ms)+1),
        "rr_ms": np.round(rr_ms, 3), "rr_s": np.round(rr_s, 3)})
    n_rr = st.number_input("Liczba wierszy tabeli RR", min_value=1,
        max_value=max(1, len(rr_df)), value=min(50, max(1, len(rr_df))), step=1)
    st.dataframe(rr_df.head(int(n_rr)), height=260, use_container_width=True)

with col_rr_tacho:
    fig_tacho = go.Figure()
    fig_tacho.add_trace(go.Scattergl(x=rr_time, y=rr_ms, mode="lines+markers",
        line=dict(color=bialy, width=2), marker=dict(color=niebieski, size=6), name="RR"))
    fig_tacho.update_layout(height=260, margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False,
        xaxis_title="Czas badania [s]", yaxis_title="Odstęp RR [ms]")
    with st.container(border=True):
        st.plotly_chart(fig_tacho, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Histogram & HRV metrics
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<p style="font-size:18px;font-weight:bold;color:#0092ff;">Histogram RR i metryki HRV</p>',
    unsafe_allow_html=True)
st.markdown(f'<hr style="margin-top:10px;height:5px;border:none;background-color:{niebieski};" />',
    unsafe_allow_html=True)

col_hist, col_metrics = st.columns([3, 2])
with col_hist:
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(x=rr_ms, nbinsx=int(liczba_binow),
        marker=dict(color=lekki_czerwony), name="RR"))
    fig_hist.update_layout(height=280, margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False,
        xaxis_title="RR [ms]", yaxis_title="Liczba wystąpień")
    with st.container(border=True):
        st.plotly_chart(fig_hist, use_container_width=True)

with col_metrics:
    m1, m2 = st.columns(2)
    m3, m4 = st.columns(2)
    m5, m6 = st.columns(2)
    with m1: st.metric("Średnie RR", f"{metrics['mean_rr']:.2f} ms")
    with m2: st.metric("SDNN",       f"{metrics['sdnn']:.2f} ms")
    with m3: st.metric("RMSSD",      f"{metrics['rmssd']:.2f} ms")
    with m4: st.metric("pNN50",      f"{metrics['pnn50']:.2f} %")
    with m5: st.metric("Min RR",     f"{metrics['min_rr']:.2f} ms")
    with m6: st.metric("Max RR",     f"{metrics['max_rr']:.2f} ms")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — EMD
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<p style="font-size:18px;font-weight:bold;color:#0092ff;">EMD i usuwanie modulacji</p>',
    unsafe_allow_html=True)
st.markdown(f'<hr style="margin-top:10px;height:5px;border:none;background-color:{niebieski};" />',
    unsafe_allow_html=True)

col_emd_left, col_emd_right = st.columns([1.7, 4.3])

with col_emd_left:
    sygnal_emd    = st.selectbox("Wybierz sygnał do EMD", ["Spoczynek", "Wysiłek"])
    df_emd_source = df_spoczynek.copy() if sygnal_emd == "Spoczynek" else df_wysilek.copy()
    start_emd = st.number_input("Początek zakresu EMD [s]",
        min_value=float(df_emd_source["czas"].min()),
        max_value=float(df_emd_source["czas"].max()),
        value=float(df_emd_source["czas"].min()), step=0.1)
    end_emd = st.number_input("Koniec zakresu EMD [s]",
        min_value=float(df_emd_source["czas"].min()),
        max_value=float(df_emd_source["czas"].max()),
        value=min(float(df_emd_source["czas"].min()) + 20, float(df_emd_source["czas"].max())),
        step=0.1)
    if end_emd < start_emd:
        st.warning("Koniec zakresu EMD musi być większy lub równy początkowi.")

# Compute outside columns so both sides share the result
df_emd = df_emd_source[
    (df_emd_source["czas"] >= start_emd) &
    (df_emd_source["czas"] <= end_emd)
].copy()

emd_result = None

if len(df_emd) < 100:
    col_emd_left.warning("Za mało danych w wybranym zakresie do EMD.")
elif end_emd <= start_emd:
    col_emd_left.warning("Koniec zakresu musi być większy niż początek.")
else:
    try:
        czas_emd = df_emd["czas"].values
        ecg_emd  = df_emd["ecg"].values
        with st.spinner("Obliczanie EMD... ⏳"):
            imf, baseline, ecg_detrended = cached_emd(ecg_emd.astype(np.float64).tobytes())
        emd_result = dict(czas_emd=czas_emd, ecg_emd=ecg_emd,
                          imf=imf, baseline=baseline, ecg_detrended=ecg_detrended)
    except Exception as e:
        col_emd_left.error(f"Błąd EMD: {e}")

with col_emd_left:
    if emd_result is not None:
        n_imf_total = emd_result["imf"].shape[1]
        imf_options = [f"IMF {i+1}" for i in range(n_imf_total)]
        default_selection = imf_options[:min(4, n_imf_total)]
        selected_imfs = st.multiselect(
            "Wybierz IMF do rekonstrukcji",
            options=imf_options,
            default=default_selection
        )
        selected_indices = [int(s.split()[1]) - 1 for s in selected_imfs]

        csv_bytes = export_ecg_txt(emd_result["czas_emd"], emd_result["ecg_detrended"])
        st.download_button(
            label="Pobierz ECG bez modulacji",
            data=csv_bytes,
            file_name=f"ECG_bez_modulacji_{sygnal_emd}.txt",
            mime="text/plain"
        )
        if selected_indices:
            ecg_recon = np.sum(emd_result["imf"][:, selected_indices], axis=1)
            csv_recon = export_ecg_txt(emd_result["czas_emd"], ecg_recon)
            st.download_button(
                label="Pobierz ECG rekonstrukcja",
                data=csv_recon,
                file_name=f"ECG_rekonstrukcja_{sygnal_emd}.txt",
                mime="text/plain"
            )
    else:
        selected_indices = []

with col_emd_right:
    if emd_result is not None:
        # Downsample all traces for the main EMD plot
        ex, ey   = downsample(emd_result["czas_emd"], emd_result["ecg_emd"])
        bx, by   = downsample(emd_result["czas_emd"], emd_result["baseline"])
        dx, dy   = downsample(emd_result["czas_emd"], emd_result["ecg_detrended"])

        fig_emd = go.Figure()
        fig_emd.add_trace(go.Scattergl(x=ex, y=ey,
            mode="lines", name="Sygnał surowy", line=dict(color=bialo_szary, width=1.5)))
        fig_emd.add_trace(go.Scattergl(x=bx, y=by,
            mode="lines", name="Modulacja / trend", line=dict(color=lekki_czerwony, width=2)))
        fig_emd.add_trace(go.Scattergl(x=dx, y=dy,
            mode="lines", name="ECG bez modulacji", line=dict(color=niebieski, width=1.5)))
        fig_emd.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            xaxis_title="Czas [s]", yaxis_title="Amplituda [mV]")
        with st.container(border=True):
            st.plotly_chart(fig_emd, use_container_width=True)

        # Individual IMF plots — downsampled, each a different color
        imf        = emd_result["imf"]
        n_show     = min(imf.shape[1], 6)
        imf_colors = ["#0092ff", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c"]

        for i in range(n_show):
            ix, iy = downsample(emd_result["czas_emd"], imf[:, i])
            fig_imf_i = go.Figure()
            fig_imf_i.add_trace(go.Scattergl(x=ix, y=iy,
                mode="lines", name=f"IMF {i+1}",
                line=dict(width=1.5, color=imf_colors[i % len(imf_colors)])))
            fig_imf_i.update_layout(
                height=160,
                margin=dict(l=0, r=0, t=24, b=0),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
                title=dict(text=f"IMF {i+1}", font=dict(size=13), x=0.01),
                xaxis_title="Czas [s]" if i == n_show - 1 else "",
                yaxis_title="Amplituda",
                xaxis=dict(showticklabels=(i == n_show - 1))
            )
            with st.container(border=True):
                st.plotly_chart(fig_imf_i, use_container_width=True)

        # Partial reconstruction plot
        if selected_indices:
            ecg_recon = np.sum(emd_result["imf"][:, selected_indices], axis=1)
            rx, ry = downsample(emd_result["czas_emd"], ecg_recon)
            ox, oy = downsample(emd_result["czas_emd"], emd_result["ecg_emd"])

            fig_recon = go.Figure()
            fig_recon.add_trace(go.Scattergl(x=ox, y=oy,
                mode="lines", name="Sygnał surowy",
                line=dict(color=bialo_szary, width=1.5)))
            fig_recon.add_trace(go.Scattergl(x=rx, y=ry,
                mode="lines", name=f"Rekonstrukcja ({', '.join(selected_imfs)})",
                line=dict(color="#2ecc71", width=2)))
            fig_recon.update_layout(
                height=280,
                margin=dict(l=0, r=0, t=30, b=0),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                title=dict(text="Rekonstrukcja z wybranych IMF", font=dict(size=13), x=0.01),
                xaxis_title="Czas [s]",
                yaxis_title="Amplituda [mV]"
            )
            with st.container(border=True):
                st.plotly_chart(fig_recon, use_container_width=True)
