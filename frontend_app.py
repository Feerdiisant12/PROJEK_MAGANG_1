import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# =================================================================================================
# BAGIAN 1: KONFIGURASI HALAMAN, STYLE, DAN FUNGSI API
# =================================================================================================

# --- Konfigurasi Halaman Utama ---
st.set_page_config(
    page_title="Dasbor Operasional AHM",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- URL Backend API ---
BACKEND_URL = "http://127.0.0.1:5001"

# --- Style Kustom (CSS) ---
st.markdown("""
<style>
    /* Kartu Metrik KPI */
    div[data-testid="metric-container"] {
       background-color: #F0F2F6; /* Warna abu-abu terang untuk light mode */
       border: 1px solid #E0E0E0;
       padding: 15px;
       border-radius: 10px;
    }
    /* Mengatur warna teks agar kontras di light mode */
    div[data-testid="metric-container"] label {
        color: #5A5A5A;
    }
</style>
""", unsafe_allow_html=True)


# --- Fungsi-fungsi untuk Memanggil API Backend ---

# --- PERBAIKAN: Fungsi pembantu untuk membersihkan nilai NaN ---
def clean_dict_for_json(d):
    """Secara rekursif membersihkan dictionary atau list dari nilai NaN/NaT."""
    if isinstance(d, dict):
        return {k: clean_dict_for_json(v) for k, v in d.items()}
    if isinstance(d, list):
        return [clean_dict_for_json(v) for v in d]
    # Menggunakan pd.isna() karena bisa menangani berbagai tipe NaN (numpy, pandas, dll)
    if pd.isna(d):
        return None # JSON mendukung null, yang dipetakan dari None di Python
    return d

@st.cache_data(ttl=60)
def load_monitoring_data():
    """Memuat data monitoring part. Mengembalikan (data, error_message)."""
    try:
        response = requests.get(f"{BACKEND_URL}/api/monitoring_data")
        if response.status_code == 200:
            return pd.DataFrame(response.json()), None
        return pd.DataFrame(), f"Error (Monitoring): {response.json().get('error', 'Gagal memuat data')}"
    except requests.exceptions.ConnectionError:
        return pd.DataFrame(), "Koneksi ke backend gagal."

@st.cache_data(ttl=120)
def load_meeting_dashboard_data():
    """Memuat data rapat pagi. Mengembalikan (data, error_message)."""
    try:
        response = requests.get(f"{BACKEND_URL}/api/morning_meeting_dashboard")
        if response.status_code == 200:
            return response.json(), None
        return None, f"Error (Rapat Pagi): {response.json().get('error', 'Gagal memuat data')}"
    except requests.exceptions.ConnectionError:
        return None, "Koneksi ke backend gagal."

def get_monitoring_insight(part_data):
    """Meminta insight AI untuk satu part dari tab monitoring."""
    try:
        # PERBAIKAN: Membersihkan data sebelum mengirim
        cleaned_part_data = clean_dict_for_json(part_data)
        response = requests.post(f"{BACKEND_URL}/api/generate_monitoring_insight", json=cleaned_part_data)
        if response.status_code == 200:
            return response.json().get("insight_text")
        return f"Gagal mendapatkan insight: {response.json().get('error')}"
    except requests.exceptions.ConnectionError:
        return "Koneksi ke backend gagal."

def get_holistic_insight(rekap_analysis):
    """Meminta insight holistik dari Gemini berdasarkan data REKAP."""
    try:
        # PERBAIKAN: Membersihkan data sebelum mengirim
        cleaned_rekap_analysis = clean_dict_for_json(rekap_analysis)
        payload = {"rekap_analysis": cleaned_rekap_analysis}
        response = requests.post(f"{BACKEND_URL}/api/generate_holistic_insight", json=payload)
        if response.status_code == 200:
            return response.json().get("insight_text")
        return f"Gagal mendapatkan insight: {response.json().get('error')}"
    except requests.exceptions.ConnectionError:
        return "Koneksi ke backend gagal."

# =================================================================================================
# BAGIAN 2: TATA LETAK UTAMA APLIKASI (HEADER DAN TABS)
# =================================================================================================

st.title("🏭 Dasbor Operasional AHM")
st.caption(f"Data diperbarui pada: {datetime.now().strftime('%d %B %Y, %H:%M:%S')}")

tab1, tab2 = st.tabs(["📊  Monitoring Part Kritis", "📈  Dasbor Rapat Pagi"])

# =================================================================================================
# BAGIAN 3: KONTEN TAB 1 - MONITORING PART KRITIS (DENGAN UPDATE)
# =================================================================================================
with tab1:
    st.header("Monitoring Prediksi Part & Anomali Stok")
    
    with st.sidebar:
        st.header("Kontrol Monitoring")
        if st.button("🔄 Refresh Data", key="refresh_monitoring", use_container_width=True):
            st.cache_data.clear()
    
    df_monitoring, error_monitoring = load_monitoring_data()
    
    if error_monitoring:
        st.error(error_monitoring)
    elif not df_monitoring.empty:
        with st.sidebar:
            st.subheader("Filter Tampilan")
            # Mengisi NaN di kolom prediksi agar tidak error saat filtering
            df_monitoring['Prediksi_Kritis'] = df_monitoring['Prediksi_Kritis'].fillna('N/A')
            df_monitoring['Status_Stok'] = df_monitoring['Status_Stok'].fillna('N/A')

            crit_options = ['Semua'] + df_monitoring['Prediksi_Kritis'].unique().tolist()
            selected_crit = st.multiselect("Tingkat Kritis", options=crit_options, default='Semua')

            anom_options = ['Semua', 'Anomali', 'Normal', 'N/A']
            selected_anom = st.radio("Status Stok", options=anom_options, horizontal=True)

            df_filtered = df_monitoring.copy()
            if 'Semua' not in selected_crit:
                df_filtered = df_filtered[df_filtered['Prediksi_Kritis'].isin(selected_crit)]
            if selected_anom != 'Semua':
                df_filtered = df_filtered[df_filtered['Status_Stok'] == selected_anom]
        
        # --- Tampilan KPI ---
        total_parts = len(df_filtered)
        high_crit_parts = len(df_filtered[df_filtered['Prediksi_Kritis'] == 'High'])
        anomaly_parts = len(df_filtered[df_filtered['Status_Stok'] == 'Anomali'])

        kpi1, kpi2, kpi3 = st.columns(3)
        with kpi1: st.metric(label="Total Parts Dimonitor", value=total_parts)
        with kpi2: st.metric(label="Parts Kritis (Prediksi 'High')", value=high_crit_parts)
        with kpi3: st.metric(label="Parts dengan Anomali Stok", value=anomaly_parts)

        st.markdown("---")

        # --- Visualisasi Baru untuk Tab 1 ---
        vis_col1, vis_col2 = st.columns([2, 1])
        with vis_col1:
            st.subheader("Peta Kondisi Stok")
            # Scatter plot untuk melihat hubungan stok dan ketahanannya
            df_filtered['Days_of_Stock'] = pd.to_numeric(df_filtered['Days_of_Stock'], errors='coerce').fillna(0)
            df_filtered['STOCK WH1'] = pd.to_numeric(df_filtered['STOCK WH1'], errors='coerce').fillna(0)
            df_filtered['WIP'] = pd.to_numeric(df_filtered['WIP'], errors='coerce').fillna(0)
            
            fig_scatter = px.scatter(
                df_filtered,
                x="STOCK WH1",
                y="Days_of_Stock",
                size="WIP",
                color="Prediksi_Kritis",
                hover_name="PART NUMBER / PART NAME",
                color_discrete_map={'High': '#D32F2F', 'Medium': '#F57C00', 'Low': '#388E3C', 'N/A': '#BDBDBD'},
                title="Stok Aktual vs. Estimasi Hari Bertahan",
                labels={"STOCK WH1": "Jumlah Stok di Gudang (Unit)", "Days_of_Stock": "Estimasi Stok Bertahan (Hari)"}
            )
            fig_scatter.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='#F0F2F6')
            st.plotly_chart(fig_scatter, use_container_width=True)

        with vis_col2:
            st.subheader("Distribusi Prediksi")
            # Bar chart untuk distribusi tingkat kritis
            crit_counts = df_filtered['Prediksi_Kritis'].value_counts()
            fig_bar = px.bar(
                crit_counts,
                x=crit_counts.index,
                y=crit_counts.values,
                color=crit_counts.index,
                color_discrete_map={'High': '#D32F2F', 'Medium': '#F57C00', 'Low': '#388E3C', 'N/A': '#BDBDBD'},
                title="Jumlah Part per Tingkat Kritis",
                labels={'x': 'Tingkat Kritis', 'y': 'Jumlah Part'}
            )
            fig_bar.update_layout(showlegend=False, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='#F0F2F6')
            st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("---")
        
        # --- Tabel Data & Insight AI ---
        
        st.subheader("Detail Status Part")
        def style_dataframe(df):
            def highlight_rows(row):
                style = [''] * len(row)
                if row['Prediksi_Kritis'] == 'High': style = ['color: #D32F2F; font-weight: bold;'] * len(row)
                elif row['Prediksi_Kritis'] == 'Medium': style = ['color: #F57C00;'] * len(row)
                if row['Status_Stok'] == 'Anomali': style = [s + ' border: 1.5px solid #D32F2F;' for s in style]
                return style
            return df.style.apply(highlight_rows, axis=1)
        st.dataframe(style_dataframe(df_filtered), use_container_width=True)

        st.markdown("---")
        st.subheader("🤖 Analisis Part oleh AI")
        part_list_monitoring = df_filtered['PART NUMBER / PART NAME'].tolist()
        if part_list_monitoring:
            selected_part_monitoring = st.selectbox("Pilih Part untuk dianalisis:", part_list_monitoring, key="monitoring_selector")
            
            if st.button("✨ Dapatkan Insight & Rekomendasi", key="insight_monitoring", use_container_width=True, type="primary"):
                with st.spinner("Gemini sedang menganalisis part yang dipilih..."):
                    part_data_monitoring = df_filtered[df_filtered['PART NUMBER / PART NAME'] == selected_part_monitoring].to_dict('records')[0]
                    insight_text = get_monitoring_insight(part_data_monitoring)
                    st.info(insight_text)
        else:
            st.warning("Tidak ada data untuk dianalisis.")

    else:
        st.warning("Tidak dapat memuat data monitoring. Cek koneksi ke backend atau isi dari Google Sheet 'template_monitoring'.")


# =================================================================================================
# BAGIAN 4: KONTEN TAB 2 - DASBOR RAPAT PAGI
# =================================================================================================
with tab2:
    st.header("Dasbor Kesehatan Stok untuk Rapat Pagi")
    
    dashboard_data, error_dashboard = load_meeting_dashboard_data()

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
                fig_bar.update_layout(title_text="Urutan Part Berdasarkan Kesehatan Stok (Order - Datang)", xaxis_title="Nilai Kesehatan Stok (Minus = Kritis)", yaxis_title="Nama Part", yaxis=dict(autorange="reversed"), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='#F0F2F6')
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.info("Data komponen kritis tidak tersedia.")

        with chart_col2:
            st.subheader("Analisis Kesehatan Stok per Tipe")
            if not summary_by_type.empty:
                fig_treemap = px.treemap(summary_by_type, path=[px.Constant("Semua Tipe"), 'TYPE'], values='Part_Count', color='Avg_Stock_Health', color_continuous_scale='RdYlGn', color_continuous_midpoint=0, title="Peta Tipe Komponen (Ukuran = Jml Part, Warna = Rata-Rata Kesehatan Stok)")
                fig_treemap.update_layout(margin = dict(t=50, l=25, r=25, b=25))
                st.plotly_chart(fig_treemap, use_container_width=True)
            else:
                st.info("Data rangkuman per tipe tidak tersedia.")
        
        st.markdown("---")
        st.subheader("🧠 Ringkasan & Rekomendasi Rapat oleh AI")
        if st.button("✨ Buat Agenda & Rekomendasi Rapat", use_container_width=True, type="primary"):
            with st.spinner("Gemini sedang menganalisis data REKAP untuk menyusun agenda rapat..."):
                insight_text = get_holistic_insight(dashboard_data)
                st.markdown(insight_text)
    else:
        st.warning("Tidak ada data untuk ditampilkan di Dasbor Rapat Pagi. Pastikan sheet 'REKAP' sudah terisi.")
