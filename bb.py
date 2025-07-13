import streamlit as st
import pandas as pd
import gspread
import joblib
import os
import google.generativeai as genai
from datetime import datetime
import numpy as np
import re
import plotly.express as px
import plotly.graph_objects as go

# =================================================================================================
# BAGIAN 1: KONFIGURASI HALAMAN DAN FUNGSI LOGIKA (YANG SEBELUMNYA DI BACKEND)
# =================================================================================================

# --- Konfigurasi Halaman Utama ---
st.set_page_config(
    page_title="Dasbor Operasional AHM",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Style Kustom (CSS) ---
st.markdown("""
<style>
    div[data-testid="metric-container"] {
       background-color: #F0F2F6;
       border: 1px solid #E0E0E0;
       padding: 15px;
       border-radius: 10px;
    }
    div[data-testid="metric-container"] label {
        color: #5A5A5A;
    }
</style>
""", unsafe_allow_html=True)

# --- Fungsi Inisialisasi dan Koneksi (Logika Backend) ---
@st.cache_resource
def initialize_gemini():
    try:
        # Mengambil API Key dari Streamlit Secrets
        api_key = st.secrets["GEMINI_API_KEY"]
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        print("✅ API Gemini berhasil dikonfigurasi.")
        return model
    except Exception as e:
        st.error(f"❌ Error konfigurasi Gemini: {e}")
        return None

@st.cache_resource
def load_models():
    try:
        model_crit = joblib.load('criticality_model.pkl')
        model_anom = joblib.load('anomaly_model.pkl')
        scaler = joblib.load('scaler.pkl')
        print("✅ Model Machine Learning berhasil dimuat.")
        return model_crit, model_anom, scaler
    except Exception as e:
        st.error(f"❌ Error memuat model .pkl: {e}")
        return None, None, None

@st.cache_resource
def connect_gspread():
    try:
        # Mengambil kredensial dari Streamlit Secrets
        creds_dict = st.secrets["gcp_service_account"]
        gc = gspread.service_account_from_dict(creds_dict)
        spreadsheet = gc.open("Monitoring part")
        ws_monitoring = spreadsheet.worksheet("template_monitoring")
        ws_rekap = spreadsheet.worksheet("REKAP")
        print("✅ Koneksi Google Sheets berhasil.")
        return ws_monitoring, ws_rekap
    except Exception as e:
        st.error(f"❌ Error koneksi Google Sheets: {e}")
        return None, None

# --- Fungsi Pengolahan Data (Logika Backend) ---
def get_clean_dataframe(worksheet):
    if not worksheet: return pd.DataFrame()
    all_values = worksheet.get_all_values()
    if not all_values or len(all_values) < 2: return pd.DataFrame()
    header = all_values[0]
    data = all_values[1:]
    df = pd.DataFrame(data, columns=header)
    df = df.loc[:, ~df.columns.isin([''])]
    df.replace('', np.nan, inplace=True)
    df.columns = [str(col).upper().strip() for col in df.columns]
    return df

@st.cache_data(ttl=60)
def load_monitoring_data(_ws_monitoring, _model_crit, _model_anom):
    if not all([_model_crit, _model_anom, _ws_monitoring]): return pd.DataFrame(), "Sistem belum siap."
    df_raw = get_clean_dataframe(_ws_monitoring)
    if df_raw.empty: return pd.DataFrame(), "Tidak ada data di sheet 'template_monitoring'."
    
    df_raw.columns = [str(col).upper().strip() for col in df_raw.columns]
    required_cols = ['STOCK WH1', 'WIP', 'PLAN']
    if not all(col in df_raw.columns for col in required_cols):
        return pd.DataFrame(), f"Kolom hilang di 'template_monitoring': {required_cols}"

    df_predict = df_raw[required_cols].copy()
    df_predict.rename(columns={'STOCK WH1': 'Stock Level', 'WIP': 'WIP', 'PLAN': 'Deviation'}, inplace=True)
    for col in df_predict.columns:
        df_predict[col] = pd.to_numeric(df_predict[col], errors='coerce')
    df_predict.dropna(inplace=True)

    if df_predict.empty: return df_raw, "Data valid untuk prediksi tidak ditemukan."

    crit_prediction = _model_crit.predict(df_predict)
    anomaly_prediction = _model_anom.predict(df_predict[['Stock Level', 'WIP']])
    anomaly_label = ['Anomali' if x == -1 else 'Normal' for x in anomaly_prediction]

    df_raw.loc[df_predict.index, 'Prediksi_Kritis'] = crit_prediction
    df_raw.loc[df_predict.index, 'Status_Stok'] = anomaly_label
    
    df_raw['PLAN_NUMERIC'] = pd.to_numeric(df_raw['PLAN'], errors='coerce')
    df_raw['STOCK_WH1_NUMERIC'] = pd.to_numeric(df_raw['STOCK WH1'], errors='coerce')
    
    daily_consumption = df_raw['PLAN_NUMERIC'] / 22 
    df_raw['Days_of_Stock'] = (df_raw['STOCK_WH1_NUMERIC'] / daily_consumption).round(1)
    
    return df_raw, None

@st.cache_data(ttl=120)
def load_meeting_dashboard_data(_ws_rekap):
    if not _ws_rekap: return None, "Koneksi ke sheet 'REKAP' gagal."
    df_rekap = get_clean_dataframe(_ws_rekap)
    if df_rekap.empty: return None, "Tidak ada data di sheet 'REKAP'."
    
    required_cols = ['PART NUMBER', 'PART NAME', 'TYPE', 'ORDER + TOLERANSI - KEDATANGAN']
    if not all(col in df_rekap.columns for col in required_cols):
        return None, f"Kolom hilang di 'REKAP': {required_cols}"

    df_rekap.rename(columns={'PART NUMBER': 'Part_Number', 'PART NAME': 'Part_Name', 'ORDER + TOLERANSI - KEDATANGAN': 'Stock_Health'}, inplace=True)
    df_rekap['Stock_Health'] = pd.to_numeric(df_rekap['Stock_Health'], errors='coerce').fillna(0)
    
    top_critical_parts = df_rekap.sort_values(by='Stock_Health', ascending=True).head(10)
    summary_by_type = df_rekap.groupby('TYPE')['Stock_Health'].agg(['mean', 'count']).reset_index()
    summary_by_type.rename(columns={'mean': 'Avg_Stock_Health', 'count': 'Part_Count'}, inplace=True)
    summary_by_type['Avg_Stock_Health'] = summary_by_type['Avg_Stock_Health'].round(2)
    summary_by_type = summary_by_type.sort_values(by='Avg_Stock_Health', ascending=True)

    kpi_summary = {
        "total_parts_rekap": len(df_rekap),
        "parts_deficit_count": len(df_rekap[df_rekap['Stock_Health'] < 0]),
        "parts_surplus_count": len(df_rekap[df_rekap['Stock_Health'] > 0])
    }
    return {
        "kpi_summary": kpi_summary,
        "top_critical_parts": top_critical_parts.to_dict(orient='records'),
        "summary_by_type": summary_by_type.to_dict(orient='records')
    }, None

def get_monitoring_insight(part_data, _gemini_model):
    if not _gemini_model: return "Model AI tidak siap."
    prompt = f"""
    Anda adalah spesialis PPIC berpengalaman. Analisis data komponen berikut:
    - Nama Part: {part_data.get('PART NUMBER / PART NAME')}
    - Stok Gudang (WH1): {part_data.get('STOCK WH1')} unit
    - WIP: {part_data.get('WIP')} unit
    - Rencana Kebutuhan: {part_data.get('PLAN')} unit
    - Prediksi Kritis: {part_data.get('Prediksi_Kritis')}
    - Status Stok: {part_data.get('Status_Stok')}
    - Estimasi Stok Bertahan: {part_data.get('Days_of_Stock')} hari
    Tugas Anda: Berikan analisis singkat, langkah mitigasi jika kritis/anomali, dan satu saran perbaikan. Gunakan markdown dan emoji.
    """
    try:
        return _gemini_model.generate_content(prompt).text
    except Exception as e:
        return f"Gagal menghubungi API Gemini: {e}"

def get_holistic_insight(rekap_analysis, _gemini_model):
    if not _gemini_model: return "Model AI tidak siap."
    prompt = f"""
    Anda adalah Manajer Operasional. Berdasarkan data kesehatan stok berikut, berikan insight untuk rapat pagi.
    - Rangkuman: {rekap_analysis.get('kpi_summary')}
    - Top 5 Kritis: {rekap_analysis.get('top_critical_parts', [])[:5]}
    - Rangkuman per Tipe: {rekap_analysis.get('summary_by_type')}
    Tugas Anda: Berikan kesimpulan umum, instruksi darurat untuk 2-3 part terkritis, analisis tipe komponen bermasalah, dan satu rekomendasi strategis. Gunakan bahasa tegas dan format markdown.
    """
    try:
        return _gemini_model.generate_content(prompt).text
    except Exception as e:
        return f"Gagal menghubungi API Gemini: {e}"

# =================================================================================================
# BAGIAN 2: TATA LETAK UTAMA APLIKASI (HEADER DAN TABS)
# =================================================================================================

# --- Inisialisasi semua koneksi dan model di awal ---
gemini_model = initialize_gemini()
model_crit, model_anom, scaler = load_models()
ws_monitoring, ws_rekap = connect_gspread()

st.title("🏭 Dasbor Operasional AHM")
st.caption(f"Data diperbarui pada: {datetime.now().strftime('%d %B %Y, %H:%M:%S')}")

tab1, tab2 = st.tabs(["📊  Monitoring Part Kritis", "📈  Dasbor Rapat Pagi"])

# =================================================================================================
# BAGIAN 3: KONTEN TAB 1 - MONITORING PART KRITIS
# =================================================================================================
with tab1:
    st.header("Monitoring Prediksi Part & Anomali Stok")
    
    with st.sidebar:
        st.header("Kontrol Monitoring")
        if st.button("🔄 Refresh Data", key="refresh_monitoring", use_container_width=True):
            st.cache_data.clear()
    
    df_monitoring, error_monitoring = load_monitoring_data(ws_monitoring, model_crit, model_anom)
    
    if error_monitoring:
        st.error(error_monitoring)
    elif not df_monitoring.empty:
        with st.sidebar:
            st.subheader("Filter Tampilan")
            df_monitoring['Prediksi_Kritis'] = df_monitoring['Prediksi_Kritis'].fillna('N/A')
            df_monitoring['Status_Stok'] = df_monitoring['Status_Stok'].fillna('N/A')
            crit_options = ['Semua'] + df_monitoring['Prediksi_Kritis'].unique().tolist()
            selected_crit = st.multiselect("Tingkat Kritis", options=crit_options, default='Semua')
            anom_options = ['Semua', 'Anomali', 'Normal', 'N/A']
            selected_anom = st.radio("Status Stok", options=anom_options, horizontal=True)

            df_filtered = df_monitoring.copy()
            if 'Semua' not in selected_crit: df_filtered = df_filtered[df_filtered['Prediksi_Kritis'].isin(selected_crit)]
            if selected_anom != 'Semua': df_filtered = df_filtered[df_filtered['Status_Stok'] == selected_anom]
        
        kpi1, kpi2, kpi3 = st.columns(3)
        with kpi1: st.metric(label="Total Parts Dimonitor", value=len(df_filtered))
        with kpi2: st.metric(label="Parts Kritis (Prediksi 'High')", value=len(df_filtered[df_filtered['Prediksi_Kritis'] == 'High']))
        with kpi3: st.metric(label="Parts dengan Anomali Stok", value=len(df_filtered[df_filtered['Status_Stok'] == 'Anomali']))

        st.markdown("---")
        vis_col1, vis_col2 = st.columns([2, 1])
        with vis_col1:
            st.subheader("Peta Kondisi Stok")
            df_filtered['Days_of_Stock'] = pd.to_numeric(df_filtered['Days_of_Stock'], errors='coerce').fillna(0)
            df_filtered['STOCK WH1'] = pd.to_numeric(df_filtered['STOCK WH1'], errors='coerce').fillna(0)
            df_filtered['WIP'] = pd.to_numeric(df_filtered['WIP'], errors='coerce').fillna(0)
            fig_scatter = px.scatter(df_filtered, x="STOCK WH1", y="Days_of_Stock", size="WIP", color="Prediksi_Kritis", hover_name="PART NUMBER / PART NAME", color_discrete_map={'High': '#D32F2F', 'Medium': '#F57C00', 'Low': '#388E3C', 'N/A': '#BDBDBD'}, title="Stok Aktual vs. Estimasi Hari Bertahan", labels={"STOCK WH1": "Stok Gudang (Unit)", "Days_of_Stock": "Estimasi Stok Bertahan (Hari)"})
            st.plotly_chart(fig_scatter, use_container_width=True)
        with vis_col2:
            st.subheader("Distribusi Prediksi")
            crit_counts = df_filtered['Prediksi_Kritis'].value_counts()
            fig_bar = px.bar(crit_counts, x=crit_counts.index, y=crit_counts.values, color=crit_counts.index, color_discrete_map={'High': '#D32F2F', 'Medium': '#F57C00', 'Low': '#388E3C', 'N/A': '#BDBDBD'}, title="Jumlah Part per Tingkat Kritis", labels={'x': 'Tingkat Kritis', 'y': 'Jumlah Part'})
            st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("---")
        st.subheader("Detail Status Part")
        def style_dataframe(df):
            def highlight_rows(row):
                style = [''] * len(row)
                if row['Prediksi_Kritis'] == 'High': style = ['color: #D32F2F; font-weight: bold;'] * len(row)
                elif row['Prediksi_Kritis'] == 'Medium': style = ['color: #F57C00;'] * len(row)
                if row['Status_Stok'] == 'Anomali': style = [s + ' border: 1.5px solid #D32F2F;' for s in style]
                return style
            return df.style.apply(highlight_rows, axis=1)
        st.dataframe(style_dataframe(df_filtered), use_container_width=True, height=400)
        st.markdown("---")
    
        st.subheader("🤖 Analisis Part oleh AI")
        part_list_monitoring = df_filtered['PART NUMBER / PART NAME'].tolist()
        if part_list_monitoring:
            selected_part = st.selectbox("Pilih Part untuk dianalisis:", part_list_monitoring, key="monitoring_selector")
            if st.button("✨ Dapatkan Insight & Rekomendasi", key="insight_monitoring", use_container_width=True, type="primary"):
                with st.spinner("Gemini sedang menganalisis part..."):
                    part_data = df_filtered[df_filtered['PART NUMBER / PART NAME'] == selected_part].to_dict('records')[0]
                    insight_text = get_monitoring_insight(part_data, gemini_model)
                    st.info(insight_text)
        else:
            st.warning("Tidak ada data untuk dianalisis.")
    else:
        st.warning("Tidak dapat memuat data monitoring.")

# =================================================================================================
# BAGIAN 4: KONTEN TAB 2 - DASBOR RAPAT PAGI
# =================================================================================================
with tab2:
    st.header("Dasbor Kesehatan Stok untuk Rapat Pagi")
    dashboard_data, error_dashboard = load_meeting_dashboard_data(ws_rekap)

    if error_dashboard:
        st.error(error_dashboard)
    elif dashboard_data:
        kpi_summary = dashboard_data.get("kpi_summary", {})
        top_critical = pd.DataFrame(dashboard_data.get("top_critical_parts", []))
        summary_by_type = pd.DataFrame(dashboard_data.get("summary_by_type", []))

        st.subheader("Ringkasan Kesehatan Stok Komponen")
        kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
        with kpi_col1: st.metric(label="Total Komponen Direkap", value=kpi_summary.get('total_parts_rekap', 0))
        with kpi_col2: st.metric(label="Komponen Defisit (< 0)", value=kpi_summary.get('parts_deficit_count', 0))
        with kpi_col3: st.metric(label="Komponen Surplus (> 0)", value=kpi_summary.get('parts_surplus_count', 0))

        st.markdown("---")
        chart_col1, chart_col2 = st.columns([3, 2])
        with chart_col1:
            st.subheader("Top 10 Komponen Paling Kritis (Defisit Stok)")
            if not top_critical.empty:
                top_critical['Color'] = top_critical['Stock_Health'].apply(lambda x: '#EF5350' if x < 0 else '#66BB6A')
                fig_bar = go.Figure(go.Bar(x=top_critical['Stock_Health'], y=top_critical['Part_Name'], orientation='h', marker_color=top_critical['Color'], text=top_critical['Stock_Health'], textposition='outside'))
                fig_bar.update_layout(title_text="Urutan Part Berdasarkan Kesehatan Stok", xaxis_title="Nilai Kesehatan Stok (Minus = Kritis)", yaxis_title="Nama Part", yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig_bar, use_container_width=True)
        with chart_col2:
            st.subheader("Analisis Kesehatan Stok per Tipe")
            if not summary_by_type.empty:
                fig_treemap = px.treemap(summary_by_type, path=[px.Constant("Semua Tipe"), 'TYPE'], values='Part_Count', color='Avg_Stock_Health', color_continuous_scale='RdYlGn', color_continuous_midpoint=0, title="Peta Tipe Komponen")
                st.plotly_chart(fig_treemap, use_container_width=True)
        
        st.markdown("---")
        st.subheader("🧠 Ringkasan & Rekomendasi Rapat oleh AI")
        if st.button("✨ Buat Agenda & Rekomendasi Rapat", use_container_width=True, type="primary"):
            with st.spinner("Gemini sedang menyusun agenda rapat..."):
                insight_text = get_holistic_insight(dashboard_data, gemini_model)
                st.markdown(insight_text)
    else:
        st.warning("Tidak ada data untuk ditampilkan di Dasbor Rapat Pagi.")
