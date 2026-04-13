# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import streamlit as st
import plotly.graph_objects as go
import emd

st.set_page_config(layout="wide", page_title="Analiza HRV")

# ── Inicjalizacja stanu aplikacji (Pamięć przycisków) ─────────────────────────
if "aktywny_sygnal" not in st.session_state:
    st.session_state["aktywny_sygnal"] = "Spoczynek"

# ── Kolory (Ciemny motyw) ─────────────────────────────────────────────────────
bialy          = "#ffffff"
bialo_szary    = "#d3d3d3"
niebieski      = "#0092ff"
ciemny_niebieski = "#005bb5"
czerwony       = "#e74c3c"
lekki_czerwony = "#e74c3c"
tlo_paneli     = "#2b2b2b"
mocny_szary    = "#444444"
czarny         = "#121212"

st.markdown(f"""
    <style>
    .stApp {{ background-color: {czarny}; color: {bialy}; }}
    h1, h2, h3, h4, h5, h6, [data-testid="stHeader"] {{ color: {bialy} !important; }}
    p, .stText, [data-testid="stWidgetLabel"] {{ color: {bialo_szary}; font-size: 15px; }}
    
    /* Pasek metryk na dole */
    .metric-row {{ border-top: 2px solid {niebieski}; border-bottom: 2px solid {niebieski}; padding: 10px 0; margin-top: 20px; }}
    [data-testid="stMetricValue"] {{ font-size: 22px !important; color: {bialy} !important; font-weight: bold; }}
    [data-testid="stMetricLabel"] p {{ color: {bialo_szary} !important; font-size: 14px; }}
    
    /* Stylizacja niebieskiego panelu z lewej strony dla parametrów R */
    div[data-testid="stVerticalBlock"] > div:first-child > div.blue-box {{
        background-color: {niebieski};
        padding: 20px;
        border-radius: 10px;
    }}
    </style>
""", unsafe_allow_html=True)

# ── Funkcje pomocnicze ────────────────────────────────────────────────────────
def downsample(x, y, max_points=3000):
    n = len(x)
    if n <= max_points: return x, y
    idx = np.round(np.linspace(0, n - 1, max_points)).astype(int)
    return x[idx], y[idx]

@st.cache_data
def load_my_data(file):
    data = pd.read_csv(file, sep='\t', decimal=',', skiprows=6,
        header=None, engine='python', encoding='cp1250', on_bad_lines='skip')
    data = data.iloc[:, :2].copy()
    data.columns = ['czas', 'ecg']
    data['czas'] = data['czas'].astype(str).str.replace(',', '.', regex=False)
    data['ecg']  = data['ecg'].astype(str).str.replace(',', '.', regex=False)
    data = data.apply(pd.to_numeric, errors='coerce').dropna().reset_index(drop=True)
    return data

@st.cache_data
def cached_emd(ecg_array):
    imf = emd.sift.sift(ecg_array)
    n_imf = imf.shape[1]
    if n_imf >= 9: baseline = imf[:, 7] + imf[:, 8]
    elif n_imf >= 8: baseline = imf[:, 6] + imf[:, 7]
    elif n_imf >= 7: baseline = imf[:, 5] + imf[:, 6]
    elif n_imf >= 2: baseline = imf[:, -1] + imf[:, -2]
    else: baseline = imf[:, -1]
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

def compute_hrv_metrics(rr_ms, peaks):
    if len(rr_ms) == 0:
        return {"mean_rr": 0, "sdnn": 0, "rmssd": 0, "pnn50": 0, "min_rr": 0, "max_rr": 0, "count": 0}
    diff_rr = np.diff(rr_ms)
    return {
        "mean_rr": np.mean(rr_ms),
        "sdnn":    np.std(rr_ms, ddof=1) if len(rr_ms) > 1 else 0,
        "rmssd":   np.sqrt(np.mean(diff_rr**2)) if len(diff_rr) > 0 else 0,
        "pnn50":   100 * np.sum(np.abs(diff_rr) > 50) / len(diff_rr) if len(diff_rr) > 0 else 0,
        "min_rr":  np.min(rr_ms),
        "max_rr":  np.max(rr_ms),
        "count":   len(peaks)
    }

def export_ecg_txt(czas, ecg_clean):
    out_df = pd.DataFrame({"czas[s]": np.round(czas, 6), "ECG": np.round(ecg_clean, 6)})
    return out_df.to_csv(sep="\t", index=False).encode("utf-8")

# ── Ładowanie danych ──────────────────────────────────────────────────────────
df_spoczynek = load_my_data("Spoczynek.txt")
df_wysilek   = load_my_data("Wysilek.txt")
df_wysilek["ecg"] = df_wysilek["ecg"] / 500

# ── Nagłówek ──────────────────────────────────────────────────────────────────
st.markdown(f"""
    <div style="background-color: {tlo_paneli}; border-radius: 8px; padding: 25px; text-align: center; margin-bottom: 20px;">
        <h2 style="color: {czerwony}; margin: 0; font-weight: bold;">Analiza HRV sygnału EKG</h2>
        <p style="color: {bialo_szary}; margin: 5px 0 0 0; font-size: 16px;">Zaawansowane laboratorium fizyki medycznej</p>
    </div>
""", unsafe_allow_html=True)

# ── PRZYCISKI DO PRZEŁĄCZANIA DANYCH (Zamiast sidebaru) ───────────────────────
st.write("") # Delikatny odstęp
c_spacer1, c_btn1, c_btn2, c_spacer2 = st.columns([3, 1, 1, 3])

with c_btn1:
    # Używamy st.rerun(), aby strona przeładowała się natychmiast i zaktualizowała kolory guzików
    if st.button("🔵 Spoczynek", type="primary" if st.session_state["aktywny_sygnal"] == "Spoczynek" else "secondary", use_container_width=True):
        st.session_state["aktywny_sygnal"] = "Spoczynek"
        st.rerun()

with c_btn2:
    if st.button("🔴 Wysiłek", type="primary" if st.session_state["aktywny_sygnal"] == "Wysiłek" else "secondary", use_container_width=True):
        st.session_state["aktywny_sygnal"] = "Wysiłek"
        st.rerun()

st.markdown("<br>", unsafe_allow_html=True)

# Przypisanie aktywnych danych na podstawie klikniętego guzika
df_active = df_spoczynek.copy() if st.session_state["aktywny_sygnal"] == "Spoczynek" else df_wysilek.copy()

# ══════════════════════════════════════════════════════════════════════════════
# SEKCJA 1 — Wybór zakresu i podgląd
# ══════════════════════════════════════════════════════════════════════════════
with st.container(border=True):
    col_tabela, col_donut, col_wykres = st.columns([1.5, 2, 5])
    
    t_min = float(df_active["czas"].min())
    t_max = float(df_active["czas"].max())
    
    with col_donut:
        zakres = st.slider("Wybierz zakres czasu do analizy [s]:", 
                           min_value=t_min, max_value=t_max, 
                           value=(t_min, min(t_min + 65.95, t_max)), step=0.1)
        start_t, end_t = zakres
        
        # Wykres kołowy %
        total_len = t_max - t_min
        sel_len = end_t - start_t
        proc = int((sel_len / total_len) * 100) if total_len > 0 else 0
        
        fig_donut = go.Figure(data=[go.Pie(
            labels=['Fragment do analizy', 'Pozostała część'],
            values=[sel_len, total_len - sel_len],
            hole=0.6,
            marker=dict(colors=[czerwony, mocny_szary]),
            textinfo='none'
        )])
        fig_donut.update_layout(
            showlegend=True, legend=dict(orientation="h", y=-0.1, x=0.1),
            margin=dict(t=20, b=0, l=0, r=0), height=220,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            annotations=[dict(text=f"{proc}%", x=0.5, y=0.5, font_size=20, showarrow=False, font_color=bialy)]
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    df_sel = df_active[(df_active["czas"] >= start_t) & (df_active["czas"] <= end_t)].copy()
    
    with col_tabela:
        st.write("") # Spacer
        st.write("")
        st.dataframe(df_sel.head(100), height=250, use_container_width=True)
        
    with col_wykres:
        fig_main = go.Figure()
        
        # Szare tło (Pozostała część) - downsampled
        x_full, y_full = downsample(df_active["czas"].values, df_active["ecg"].values)
        fig_main.add_trace(go.Scatter(x=x_full, y=y_full, mode="lines", name="Pozostała część", line=dict(color=mocny_szary, width=1)))
        
        # Czerwony fragment (Wybrany) - downsampled
        x_sel, y_sel = downsample(df_sel["czas"].values, df_sel["ecg"].values)
        fig_main.add_trace(go.Scatter(x=x_sel, y=y_sel, mode="lines", name="Fragment do analizy", line=dict(color=czerwony, width=1.5)))
        
        fig_main.update_layout(
            height=280, margin=dict(l=0, r=0, t=30, b=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            xaxis_title="Czas [s]", yaxis_title="Amplituda [mV]",
            xaxis=dict(showgrid=False), yaxis=dict(showgrid=False)
        )
        st.plotly_chart(fig_main, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# SEKCJA 2 — Analiza RR i Histogram
# ══════════════════════════════════════════════════════════════════════════════
col_rr_main, col_hist_main = st.columns(2)

# ---- LEWA STRONA: RR ----
with col_rr_main:
    st.markdown('<p style="font-size:18px;font-weight:bold;color:#0092ff;">Identyfikacja załamków R i tworzenie szeregu RR</p>', unsafe_allow_html=True)
    
    c_params, c_plots = st.columns([1, 2.5])
    with c_params:
        st.markdown(f'<div class="blue-box" style="background-color: {niebieski}; padding: 15px; border-radius: 8px; margin-top: 25px;">', unsafe_allow_html=True)
        prog_r = st.slider("Próg dla pików R:", 
                           min_value=float(df_sel["ecg"].min()), 
                           max_value=float(df_sel["ecg"].max()), 
                           value=float(df_sel["ecg"].max()) * 0.5, step=0.01)
        dystans_r = st.slider("Dystans między RR:", 
                              min_value=200.0, max_value=1200.0, value=450.0, step=10.0)
        st.markdown('</div>', unsafe_allow_html=True)
        
    peaks, _ = detect_r_peaks(df_sel, sampling_rate=1000, distance_ms=int(dystans_r), height=prog_r)
    rr_s, rr_ms, rr_time, r_times = compute_rr(df_sel["czas"], peaks)
    metrics = compute_hrv_metrics(rr_ms, peaks)
    
    with c_plots:
        # Wykres EKG z pikami R
        fig_r = go.Figure()
        x_r, y_r = downsample(df_sel["czas"].values, df_sel["ecg"].values)
        fig_r.add_trace(go.Scatter(x=x_r, y=y_r, mode="lines", name="Sygnał EKG", line=dict(color=bialo_szary, width=1.2)))
        if len(peaks) > 0:
            fig_r.add_trace(go.Scatter(x=df_sel["czas"].iloc[peaks], y=df_sel["ecg"].iloc[peaks], 
                                       mode="markers", name="Piki R", marker=dict(color=niebieski, size=6)))
        fig_r.update_layout(height=180, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                            xaxis_title="Czas [s]", yaxis_title="Amplituda [mV]")
        st.plotly_chart(fig_r, use_container_width=True)
        
        # Wykres Tachygram
        fig_tach = go.Figure()
        fig_tach.add_trace(go.Scatter(x=rr_time, y=rr_ms, mode="lines+markers", line=dict(color=bialy, width=2), marker=dict(color=niebieski, size=5)))
        fig_tach.update_layout(height=200, margin=dict(l=0, r=0, t=20, b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False,
                               xaxis_title="Czas badania [s]", yaxis_title="Odstęp RR [ms]")
        st.plotly_chart(fig_tach, use_container_width=True)

# ---- PRAWA STRONA: Histogram i Metryki ----
with col_hist_main:
    st.markdown('<p style="font-size:18px;font-weight:bold;color:#0092ff;">Histogram</p>', unsafe_allow_html=True)
    
    with st.container(border=True): # Obejma dla histogramu i tabeli
        c_tab, c_hist = st.columns([1, 1.8])
        
        with c_tab:
            rr_df = pd.DataFrame({"rr_ms": np.round(rr_ms, 0), "rr_s": np.round(rr_s, 3)})
            st.dataframe(rr_df, height=250, use_container_width=True)
            
        with c_hist:
            liczba_binow = st.slider("Liczba przedziałów", min_value=10, max_value=200, value=180, step=10, label_visibility="collapsed")
            fig_h = go.Figure()
            fig_h.add_trace(go.Histogram(x=rr_ms, nbinsx=int(liczba_binow), marker=dict(color=niebieski)))
            fig_h.update_layout(height=220, margin=dict(l=0, r=0, t=20, b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                xaxis_title="Czas trwania [ms]", yaxis_title="Częstość", bargap=0.1)
            st.plotly_chart(fig_h, use_container_width=True)
            
    # Metryki na dole prawej strony
    st.markdown(f'<div class="metric-row">', unsafe_allow_html=True)
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1: st.metric("Średnie RR", f"{int(metrics['mean_rr'])} ms")
    with m2: st.metric("Std RR", f"{int(metrics['sdnn'])} ms")
    with m3: st.metric("Max RR", f"{int(metrics['max_rr'])} ms")
    with m4: st.metric("Min RR", f"{int(metrics['min_rr'])} ms")
    with m5: st.metric("Liczba załamków R", f"{metrics['count']}")
    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — EMD 
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown('<p style="font-size:18px;font-weight:bold;color:#0092ff;">EMD i usuwanie modulacji</p>', unsafe_allow_html=True)
st.markdown(f'<hr style="margin-top:0px;height:2px;border:none;background-color:{niebieski};" />', unsafe_allow_html=True)

col_emd_left, col_emd_right = st.columns([1.7, 4.3])

with col_emd_left:
    start_emd = st.number_input("Początek zakresu EMD [s]",
        min_value=float(df_active["czas"].min()), max_value=float(df_active["czas"].max()),
        value=start_t, step=0.1)
    end_emd = st.number_input("Koniec zakresu EMD [s]",
        min_value=float(df_active["czas"].min()), max_value=float(df_active["czas"].max()),
        value=end_t, step=0.1)
    if end_emd < start_emd:
        st.warning("Koniec zakresu EMD musi być większy lub równy początkowi.")

df_emd = df_active[(df_active["czas"] >= start_emd) & (df_active["czas"] <= end_emd)].copy()
emd_result = None

if len(df_emd) < 100:
    col_emd_left.warning("Za mało danych w wybranym zakresie do EMD.")
else:
    try:
        czas_emd = df_emd["czas"].values
        ecg_emd  = df_emd["ecg"].values
        imf, baseline, ecg_detrended = cached_emd(ecg_emd)
        emd_result = dict(czas_emd=czas_emd, ecg_emd=ecg_emd, imf=imf, baseline=baseline, ecg_detrended=ecg_detrended)
    except Exception as e:
        col_emd_left.error(f"Błąd EMD: {e}")

with col_emd_left:
    if emd_result is not None:
        n_imf_total = emd_result["imf"].shape[1]
        imf_options = [f"IMF {i+1}" for i in range(n_imf_total)]
        default_selection = imf_options[:min(4, n_imf_total)]
        selected_imfs = st.multiselect("Wybierz IMF do rekonstrukcji", options=imf_options, default=default_selection)
        selected_indices = [int(s.split()[1]) - 1 for s in selected_imfs]

        csv_bytes = export_ecg_txt(emd_result["czas_emd"], emd_result["ecg_detrended"])
        st.download_button(label="Pobierz ECG bez modulacji", data=csv_bytes, file_name=f"ECG_bez_modulacji.txt", mime="text/plain")
        if selected_indices:
            ecg_recon = np.sum(emd_result["imf"][:, selected_indices], axis=1)
            csv_recon = export_ecg_txt(emd_result["czas_emd"], ecg_recon)
            st.download_button(label="Pobierz ECG rekonstrukcja", data=csv_recon, file_name=f"ECG_rekonstrukcja.txt", mime="text/plain")
    else:
        selected_indices = []

with col_emd_right:
    if emd_result is not None:
        # Rekonstrukcja
        if selected_indices:
            ecg_recon = np.sum(emd_result["imf"][:, selected_indices], axis=1)
            rx, ry = downsample(emd_result["czas_emd"], ecg_recon)
            ox, oy = downsample(emd_result["czas_emd"], emd_result["ecg_emd"])

            fig_recon = go.Figure()
            fig_recon.add_trace(go.Scatter(x=ox, y=oy, mode="lines", name="Sygnał surowy", line=dict(color=bialo_szary, width=1.5)))
            fig_recon.add_trace(go.Scatter(x=rx, y=ry, mode="lines", name=f"Rekonstrukcja ({', '.join(selected_imfs)})", line=dict(color="#2ecc71", width=2)))
            fig_recon.update_layout(height=280, margin=dict(l=0, r=0, t=30, b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                title=dict(text="Rekonstrukcja z wybranych IMF", font=dict(size=13), x=0.80, y=1.0),
                xaxis_title="Czas [s]", yaxis_title="Amplituda [mV]")
            with st.container(border=True):
                st.plotly_chart(fig_recon, use_container_width=True)
                
        # Główny EMD
        ex, ey   = downsample(emd_result["czas_emd"], emd_result["ecg_emd"])
        bx, by   = downsample(emd_result["czas_emd"], emd_result["baseline"])
        dx, dy   = downsample(emd_result["czas_emd"], emd_result["ecg_detrended"])

        fig_emd = go.Figure()
        fig_emd.add_trace(go.Scatter(x=ex, y=ey, mode="lines", name="Sygnał surowy", line=dict(color=bialo_szary, width=1.5)))
        fig_emd.add_trace(go.Scatter(x=bx, y=by, mode="lines", name="Modulacja / trend", line=dict(color=lekki_czerwony, width=2)))
        fig_emd.add_trace(go.Scatter(x=dx, y=dy, mode="lines", name="ECG bez modulacji", line=dict(color=niebieski, width=1.5)))
        fig_emd.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            xaxis_title="Czas [s]", yaxis_title="Amplituda [mV]")
        with st.container(border=True):
            st.plotly_chart(fig_emd, use_container_width=True)

        # Składowe IMF
        imf        = emd_result["imf"]
        n_show     = min(imf.shape[1], 9)
        imf_colors = ["#0092ff", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c"]

        for i in range(n_show):
            ix, iy = downsample(emd_result["czas_emd"], imf[:, i])
            fig_imf_i = go.Figure()
            fig_imf_i.add_trace(go.Scatter(x=ix, y=iy, mode="lines", name=f"IMF {i+1}", line=dict(width=1.5, color=imf_colors[i % len(imf_colors)])))
            fig_imf_i.update_layout(height=160, margin=dict(l=0, r=0, t=24, b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False,
                title=dict(text=f"IMF {i+1}", font=dict(size=13), x=0.01),
                xaxis_title="Czas [s]" if i == n_show - 1 else "", yaxis_title="Amplituda", xaxis=dict(showticklabels=(i == n_show - 1)))
            with st.container(border=True):
                st.plotly_chart(fig_imf_i, use_container_width=True)