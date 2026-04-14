# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
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
    p, .stText {{ color: {bialo_szary}; font-size: 16px; }}
    
    .moja-ramka {{ 
        border-radius:10px; padding:20px; background-color:{czerwony};
        text-align:center; margin-bottom: 20px;
    }}
    .moja-ramka h4 {{ color:{czarny} !important; margin:0; font-weight: bold; }}
    
    /* Stylizacja metryk z kodu ziomka */
    [data-testid="stMetricValue"] {{ font-size: 18px !important; color: {bialy} !important; }}
    [data-testid="stMetricLabel"] p {{ color: {mocny_szary} !important; }}
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
    url = f'https://drive.google.com/uc?export=download&id={file_id}'
    try:
        response = requests.get(url)
        response.raise_for_status()
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
st.markdown(f'<div class="moja-ramka"><h4>Analiza HRV sygnału EKG</h4><p style="color:{czarny};">laboratorium fizyki medycznej</p></div>', unsafe_allow_html=True)

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
# SECTION 2 — Identyfikacja załamków R (Nowy design)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""<hr style="margin-top: 10px;height:5px; border:none; color:#444444; background-color:#444444;" />""", unsafe_allow_html=True)

# Wybór sygnału dla sekcji RR i EMD
syg_col1, syg_col2, syg_col3 = st.columns([2, 1, 1])
with syg_col1:
    sygnal = st.selectbox("Sygnał do analizy szczegółowej (RR i EMD):", ["Spoczynek", "Wysiłek"])
    df_active = df_spoczynek if sygnal == "Spoczynek" else df_wysilek
with syg_col2:
    start_rr = st.number_input("Start analizy [s]", value=0.0)
with syg_col3:
    end_rr = st.number_input("Koniec analizy [s]", value=60.0)

df_rr_signal = df_active[(df_active["czas"] >= start_rr) & (df_active["czas"] <= end_rr)].copy()

col1, col2 = st.columns([4, 4.5])

with col1:
    st.markdown(f'<p style="margin-top: 0px; font-size: 18px; font-weight: bold; color: {niebieski};"> Identyfikacja załamków R i tworzenie szeregu RR</p>', unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 4])

    with col_left:    
        st.markdown(f"""
            <div style="background-color: {niebieski}; 
                border-radius: 10px; 
                padding: 40px;
                margin-bottom: -310px; /* Dopasowane z -1820px */
                height: 230px;
                border: 0px solid rgba(100,100,100,1);
            ">
            </div>
        """, unsafe_allow_html=True)  
            
        lewy, srodek, prawy = st.columns([0.1, 0.9, 0.1])   
        with srodek:
            st.markdown('<div style="margin-top: 20px;"></div>', unsafe_allow_html=True)
            threshold_rr = st.slider("Próg dla pików R:", min_value=0.0, max_value=2.0, value=0.11, step=0.01)
            distance_rr = st.slider("Dystans między RR:", min_value=0.0, max_value=2000.0, value=450.0, step=10.0)

        sygnal_ecg = df_rr_signal['ecg'].values
        # Rzutowanie distance na int, bo find_peaks wymaga liczby próbek
        peaks, _ = find_peaks(sygnal_ecg, distance=int(distance_rr), height=threshold_rr)

    with col_right:    
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_rr_signal['czas'], y=df_rr_signal['ecg'], mode='lines', name='Sygnał EKG',
                                 line=dict(color='#C3E5FF', width=1.5)))
        fig.add_trace(go.Scatter(
            x=df_rr_signal['czas'].iloc[peaks] if len(peaks) > 0 else [],
            y=df_rr_signal['ecg'].iloc[peaks] if len(peaks) > 0 else [],
            mode='markers', name='Piki R',
            marker=dict(color=niebieski, size=8, symbol='circle', line=dict(color='white', width=1))
        ))
        fig.add_hline(y=threshold_rr, line_dash="dash", line_color="rgba(255,255,255,0.3)", annotation_text="Aktualny próg")

        fig.update_layout(
            height=200, margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            xaxis_title="Czas [s]", yaxis_title="Amplituda [mV]"
        )

        with st.container(border=True):
            st.plotly_chart(fig, use_container_width=True)

    # Obliczenia RR
    czasy_pikow = df_rr_signal['czas'].iloc[peaks].values if len(peaks) > 0 else np.array([])
    odstepy_rr = np.diff(czasy_pikow)
    
    df_rr = pd.DataFrame({
        '#': range(1, len(odstepy_rr) + 1),
        'rr_ms': odstepy_rr * 1000, 
        'rr_s': odstepy_rr          
    })

    st.markdown(f"""
        <div style="background-color: {lekki_szary}; 
            border-radius: 10px; 
            padding: 40px;
            margin-bottom: -380px; /* Dopasowane z -1820px */
            height: 300px;
            border: 0px solid rgba(100,100,100,1);
        ">
        </div>
    """, unsafe_allow_html=True)  
    
    lewy, srodek, prawy = st.columns([0.02, 0.9, 0.02])

    with srodek:
        fig_rr = go.Figure()
        if not df_rr.empty:
            fig_rr.add_trace(go.Scatter(
                x=czasy_pikow[1:],
                y=df_rr['rr_ms'].values,
                mode='lines+markers', name='Odstępy RR',
                line=dict(color=bialy, width=2), 
                marker=dict(size=6, color=niebieski, symbol='circle') 
            ))

        fig_rr.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            xaxis_title="Czas badania [s]", yaxis_title="Odstęp RR [ms]",
            template="plotly_dark", hovermode="x unified",
            margin=dict(l=30, r=20, t=30, b=90), height=300
        )
        st.plotly_chart(fig_rr, use_container_width=True)
    
with col2:
    st.markdown(f'<p style="margin-top: 0px; font-size: 18px; font-weight: bold; color:{niebieski};">Histogram</p>', unsafe_allow_html=True)

    st.markdown(f"""
        <div style="background-color: {lekki_szary}; 
            border-radius: 10px; 
            padding: 40px;
            margin-bottom: -630px; /* Dopasowane z -1820px */
            height: 550px;
            border: 0px solid rgba(100,100,100,1);
        ">
        </div>
    """, unsafe_allow_html=True)  
    
    lewy, srodek, prawy = st.columns([0.02, 0.9, 0.02])  
    
    with srodek:
        col_rr1, col_rr2 = st.columns([1., 1.8])

        with col_rr1:
            st.dataframe(df_rr, height=310, use_container_width=True)

        with col_rr2:
            histogram_bins = st.slider('Histogram', min_value=20, max_value=300, value=180, step=1)
            
            if not df_rr.empty:
                fig_hist = px.histogram(
                    df_rr, x="rr_ms", nbins=histogram_bins, 
                    labels={'rr_ms': 'Odstęp RR [ms]'},
                    color_discrete_sequence=[niebieski], marginal="rug" 
                )
                fig_hist.update_layout(
                    height=250, margin=dict(l=0, r=0, t=0, b=0),
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    xaxis_title="Czas trwania  [ms]", yaxis_title="Częstość", bargap=0.1 
                )
                with st.container(border=True):
                    st.plotly_chart(fig_hist, use_container_width=True)
            else:
                st.info("Zbyt mało załamków R do wygenerowania histogramu.")
    
        if not df_rr.empty:
            srednie_rr = df_rr['rr_ms'].mean()
            sdnn = df_rr['rr_ms'].std()
            max_rr = df_rr['rr_ms'].max()
            min_rr = df_rr['rr_ms'].min()
            liczba_R = df_rr.shape[0]
        else:
            srednie_rr = sdnn = max_rr = min_rr = liczba_R = 0
        
        st.markdown(f'<hr style="margin-top: 10px;height:5px; border:none; color: {niebieski}; background-color:{niebieski};" />', unsafe_allow_html=True)
    
        cola, colb, colc, cold, cole = st.columns([1,1,1,1,2])
        cola.metric("Średnie RR", f"{srednie_rr:.0f} ms")
        colb.metric("Std RR", f"{sdnn:.0f} ms")
        colc.metric("Max RR", f"{max_rr:.0f} ms")        
        cold.metric("Min RR", f"{min_rr:.0f} ms")
        cole.metric("Liczba zidentyfikowanych załamków R", f"{liczba_R:.0f}")
    
        st.markdown(f'<hr style="margin-top: 10px;height:5px; border:none; color: {niebieski}; background-color:{niebieski};" />', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — EMD (ROZŁOŻENIE NA SKŁADOWE)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f'<p style="font-size:18px;font-weight:bold;color:{niebieski};">Dekompozycja EMD (Częstotliwości)</p>', unsafe_allow_html=True)
col_emd_l, col_emd_r = st.columns([1.7, 4.3])

# Ograniczenie do 15s zaczynając od start_rr
df_emd = df_active[(df_active["czas"] >= start_rr) & (df_active["czas"] <= start_rr + 15)].copy()

if len(df_emd) > 100:
    with st.spinner("Przetwarzanie EMD..."):
        imfs, trend, _ = cached_emd(df_emd["ecg"].values.astype(np.float64).tobytes())
    
    with col_emd_l:
        st.write("Opcje rekonstrukcji EMD:")
        
        # --- NOWOŚĆ: Automatyczny wybór ---
        auto_mode = st.toggle("🤖 Wybierz składowe automatycznie", value=True)
        
        imf_ops = [f"IMF {i+1}" for i in range(imfs.shape[1])]
        n_imfs = imfs.shape[1]
        
        # Logika automatu
        if auto_mode:
            # Odrzucamy 1. składową (szum) i 2 ostatnie (pływanie linii)
            if n_imfs > 3:
                auto_default = [f"IMF {i+1}" for i in range(1, n_imfs - 2)]
            else:
                auto_default = imf_ops # Zabezpieczenie jak sygnał jest za prosty
                
            selected = st.multiselect("Wybrane składowe:", imf_ops, default=auto_default, disabled=True)
            st.caption("Tryb auto: Odrzucono skrajne częstotliwości (szum i pływanie linii).")
        else:
            selected = st.multiselect("Wybrane składowe:", imf_ops, default=imf_ops[:3])
            st.caption("Tryb ręczny: Wybierz samodzielnie.")
        
        wybrane_indeksy = [int(s.replace("IMF ", "")) - 1 for s in selected]
        
        if wybrane_indeksy:
            zrekonstruowany_sygnal = np.sum(imfs[:, wybrane_indeksy], axis=1)
        else:
            zrekonstruowany_sygnal = np.zeros(len(df_emd)) 
            
        st.download_button(
            "Pobierz czyste EKG", 
            data=export_ecg_txt(df_emd["czas"].values, zrekonstruowany_sygnal), 
            file_name="EKG_EMD_Rekonstrukcja.txt"
        )

    with col_emd_r:
        fig_emd_main = go.Figure()
        ex, ey = downsample(df_emd["czas"].values, df_emd["ecg"].values)
        _, et = downsample(df_emd["czas"].values, trend)
        _, ec = downsample(df_emd["czas"].values, zrekonstruowany_sygnal)
        
        fig_emd_main.add_trace(go.Scatter(x=ex, y=ey, name="Surowy", line=dict(color=mocny_szary, width=1)))
        fig_emd_main.add_trace(go.Scatter(x=ex, y=et, name="Trend (Niska f)", line=dict(color=lekki_czerwony, width=2)))
        fig_emd_main.add_trace(go.Scatter(x=ex, y=ec, name="Oczyszczony (Rekonstrukcja)", line=dict(color=niebieski, width=1.5)))
        
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
    st.info("Wybierz poprawny zakres czasu w panelu wyżej, aby uruchomić EMD.")
