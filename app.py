# app.py

import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import requests  # Library untuk mengambil data dari API
from datetime import datetime, timedelta

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="Dashboard PPIC - AHM",
    page_icon="🏭",
    layout="wide"
)

# --- Fungsi untuk Mengambil Data dari API ---
@st.cache_data(ttl=3600)  # Cache data selama 1 jam
def get_public_holidays():
    """Mengambil data hari libur nasional dari API publik."""
    try:
        response = requests.get("https://api-harilibur.vercel.app/api")
        if response.status_code == 200:
            holidays = response.json()
            # Filter hanya hari libur nasional, bukan cuti bersama
            national_holidays = [h for h in holidays if h['is_national_holiday']]
            return {pd.to_datetime(h['holiday_date']): h['holiday_name'] for h in national_holidays}
        return {}
    except requests.exceptions.RequestException:
        return {}

# --- Fungsi Backend Lainnya ---
@st.cache_resource
def load_artifacts():
    """Memuat model, peta konsumsi, dan data."""
    try:
        model = joblib.load('model_pipeline.pkl')
        konsumsi_map = joblib.load('konsumsi_map.pkl')
        df_raw = pd.read_csv('dummy_dataset.csv')
        df_raw.columns = df_raw.columns.str.strip()
        df_raw['tanggal'] = pd.to_datetime(df_raw['tanggal'])
        df_processed = pd.read_csv('data_preprocessed.csv')
        return model, konsumsi_map, df_raw, df_processed
    except FileNotFoundError:
        return None, None, None, None

def process_dashboard_data(df, model):
    dashboard_df = df.copy()
    dashboard_df['waktu_habis_stok'] = dashboard_df['stok_tersedia'] / dashboard_df['konsumsi_per_jam']
    dashboard_df['selisih_waktu'] = dashboard_df['waktu_habis_stok'] - dashboard_df['lead_time']
    def get_status(row):
        if row['selisih_waktu'] <= 0: return 'Merah'
        input_data = pd.DataFrame([{'seksi_tujuan': row['seksi_tujuan'], 'nama_komponen': row['nama_komponen'],
                                    'stok_tersedia': row['stok_tersedia'], 'lead_time': row['lead_time']}])
        pred = model.predict(input_data)[0]
        return 'Kuning' if pred == 'Merah' else pred
    dashboard_df['Status'] = dashboard_df.apply(get_status, axis=1)
    result_df = dashboard_df[['tanggal', 'seksi_tujuan', 'nama_komponen', 'Status', 'stok_tersedia', 
                              'waktu_habis_stok', 'lead_time', 'selisih_waktu']].rename(columns={
        'tanggal': 'Tanggal', 'seksi_tujuan': 'Seksi Tujuan', 'nama_komponen': 'Nama Komponen',
        'stok_tersedia': 'Stok', 'waktu_habis_stok': 'Sisa Waktu (Jam)',
        'lead_time': 'Lead Time (Jam)', 'selisih_waktu': 'Buffer Waktu (Jam)'
    })
    return result_df.round(1)

def style_dashboard_text_color(df):
    def color_status(val):
        color = '#E74C3C' if val == 'Merah' else '#F1C40F' if val == 'Kuning' else '#2ECC71'
        return f'color: {color}; font-weight: bold;'
    return df.style.applymap(color_status, subset=['Status'])

# --- Memuat Artefak dan Data API ---
model, konsumsi_map, df_raw, df_processed = load_artifacts()
holidays_data = get_public_holidays()

if model is None:
    st.error("File model/data tidak ditemukan! Jalankan notebook pelatihan terlebih dahulu.")
    st.stop()

# --- Tampilan Utama ---
st.title("🏭 Dashboard Pemantauan Material PPIC")
st.markdown("Sistem Peringatan Dini untuk Mencegah Potensi Downtime Lini Produksi.")

# --- FITUR BARU: Notifikasi Hari Libur ---
st.markdown("---")
with st.container():
    st.subheader("🗓️ Notifikasi Hari Libur Nasional Terdekat")
    today = datetime.now().date()
    next_7_days = [today + timedelta(days=i) for i in range(8)]
    upcoming_holidays = {date.strftime('%Y-%m-%d'): name for date, name in holidays_data.items() if date.date() in next_7_days}
    
    if upcoming_holidays:
        for date_str, name in upcoming_holidays.items():
            st.warning(f"**Perhatian:** Hari **{name}** akan berlangsung pada tanggal **{date_str}**. Harap sesuaikan perencanaan produksi dan pengiriman.", icon="🗓️")
    else:
        st.success("Tidak ada hari libur nasional dalam 7 hari ke depan.", icon="✅")
st.markdown("---")


tab1, tab2 = st.tabs(["📊 Dashboard Pemantauan Stok", " Simulasi & Analisis Detail"])

with tab1:
    st.header("Ringkasan Risiko Material di Seluruh Plant")
    col_filter1, col_filter2 = st.columns([1, 2])
    with col_filter1:
        unique_dates = df_raw['tanggal'].dt.date.unique()
        selected_date = st.selectbox("Pilih Tanggal", options=unique_dates, index=len(unique_dates)-1)
    with col_filter2:
        status_options = ['Merah', 'Kuning', 'Hijau']
        selected_statuses = st.multiselect("Filter Berdasarkan Status", options=status_options, default=status_options)

    data_on_date = df_raw[df_raw['tanggal'].dt.date == selected_date]
    
    if data_on_date.empty:
        st.warning(f"Tidak ada data produksi untuk tanggal {selected_date.strftime('%Y-%m-%d')}.")
    else:
        dashboard_data = process_dashboard_data(data_on_date, model)
        filtered_dashboard = dashboard_data[dashboard_data['Status'].isin(selected_statuses)]
        merah_count = filtered_dashboard[filtered_dashboard['Status'] == 'Merah'].shape[0]
        kuning_count = filtered_dashboard[filtered_dashboard['Status'] == 'Kuning'].shape[0]
        total_count = filtered_dashboard.shape[0]

        col1, col2, col3 = st.columns(3)
        col1.metric("🚨 Komponen Kritis (Merah)", f"{merah_count}")
        col2.metric("⚠️ Komponen Waspada (Kuning)", f"{kuning_count}")
        col3.metric("✅ Total Komponen Ditampilkan", f"{total_count}")
        
        col_data1, col_data2 = st.columns([2, 1.5])
        with col_data1:
            st.write("**Daftar Material Berdasarkan Tingkat Kekritisan:**")
            if not filtered_dashboard.empty:
                filtered_dashboard['SortKey'] = filtered_dashboard['Status'].map({'Merah': 0, 'Kuning': 1, 'Hijau': 2})
                sorted_df = filtered_dashboard.sort_values(by=['SortKey', 'Buffer Waktu (Jam)']).drop(columns=['SortKey'])
                styled_df = style_dashboard_text_color(sorted_df)
                st.dataframe(styled_df, use_container_width=True, height=500)
            else:
                st.info("Tidak ada data yang sesuai dengan filter yang dipilih.")
        with col_data2:
            st.write("**Seksi Tujuan Paling Kritis (Merah & Kuning):**")
            if not filtered_dashboard.empty:
                hotspot_data = filtered_dashboard[filtered_dashboard['Status'].isin(['Merah', 'Kuning'])]
                if not hotspot_data.empty:
                    hotspot_count = hotspot_data.groupby('Seksi Tujuan')['Status'].count().sort_values(ascending=True)
                    fig_hotspot = go.Figure(go.Bar(y=hotspot_count.index, x=hotspot_count.values, orientation='h', marker_color='#C0392B'))
                    fig_hotspot.update_layout(height=500, title_text="Jumlah Item Kritis per Seksi", xaxis_title="Jumlah Item", yaxis_title="Seksi Tujuan")
                    st.plotly_chart(fig_hotspot, use_container_width=True)
                else:
                    st.success("Tidak ada komponen berstatus Merah atau Kuning pada tanggal ini.")

with tab2:
    st.header("Analisis dan Simulasi per Komponen")
    with st.sidebar:
        st.title("⚙️ Analisis Detail")
        list_seksi = df_processed['seksi_tujuan'].unique()
        list_komponen = df_processed['nama_komponen'].unique()
        selected_seksi = st.selectbox("Pilih Seksi Tujuan", options=sorted(list_seksi), key="sb_seksi")
        selected_komponen = st.selectbox("Pilih Nama Komponen", options=sorted(list_komponen), key="sb_komponen")
        input_stok = st.number_input("Simulasi Stok Tersedia (unit)", min_value=0, value=100, step=10, key="ni_stok")
        input_lead_time = st.number_input("Simulasi Lead Time (jam)", min_value=0.0, value=2.0, step=0.5, key="ni_leadtime")
        predict_button = st.button("Jalankan Analisis", type="primary", use_container_width=True)

    if predict_button:
        konsumsi_per_jam = konsumsi_map.get((selected_seksi, selected_komponen))
        if konsumsi_per_jam is None:
            st.error(f"Kombinasi Seksi '{selected_seksi}' dan Komponen '{selected_komponen}' tidak ditemukan.")
        else:
            waktu_habis_stok = input_stok / konsumsi_per_jam if konsumsi_per_jam > 0 else float('inf')
            selisih_waktu = waktu_habis_stok - input_lead_time
            
            if selisih_waktu <= 0: final_prediksi = 'Merah'
            else:
                input_data = pd.DataFrame([{'seksi_tujuan': selected_seksi, 'nama_komponen': selected_komponen,
                                            'stok_tersedia': input_stok, 'lead_time': input_lead_time}])
                final_prediksi = model.predict(input_data)[0]
                if final_prediksi == 'Merah': final_prediksi = 'Kuning'
            
            st.subheader(f"Hasil Analisis untuk: {selected_komponen}")
            
            if final_prediksi == 'Hijau': st.success(f"**Status: HIJAU (Aman)** ✅", icon="✅")
            elif final_prediksi == 'Kuning': st.warning(f"**Status: KUNING (Waspada)** ⚠️", icon="⚠️")
            else: st.error(f"**Status: MERAH (Bahaya)** 🚨", icon="🚨")
            
            saran = f"RISIKO DOWNTIME! Stok akan habis sebelum material tiba. Segera hubungi seksi pengirim untuk percepat pengiriman **{selected_komponen}**." if final_prediksi == 'Merah' else "Waspada! Stok berpotensi menipis. Koordinasikan dengan seksi pengirim." if final_prediksi == 'Kuning' else "Stok material dalam kondisi aman. Lanjutkan pemantauan rutin."
            st.info(f"**Saran Tindakan:** {saran}", icon="💡")
            
            col_viz1, col_viz2 = st.columns(2)
            with col_viz1:
                st.subheader("Visualisasi Risiko")
                fig_bar = go.Figure(go.Bar(x=[waktu_habis_stok, input_lead_time], y=['Waktu Stok Habis', 'Lead Time Pengiriman'],
                                           orientation='h', marker_color=['#1f77b4', '#d62728'],
                                           text=[f"{waktu_habis_stok:.1f} jam", f"{input_lead_time:.1f} jam"], textposition='auto'))
                fig_bar.update_layout(title_text='Perbandingan Waktu Habis Stok vs. Lead Time', xaxis_title="Waktu (jam)", showlegend=False)
                st.plotly_chart(fig_bar, use_container_width=True)

            with col_viz2:
                st.subheader("Tren Historis Buffer Waktu")
                history_df = df_raw[(df_raw['seksi_tujuan'] == selected_seksi) & (df_raw['nama_komponen'] == selected_komponen)].copy()
                if not history_df.empty:
                    history_df['buffer_hist'] = (history_df['stok_tersedia'] / history_df['konsumsi_per_jam']) - history_df['lead_time']
                    fig_trend = go.Figure(go.Scatter(x=history_df['tanggal'], y=history_df['buffer_hist'], mode='lines+markers', name='Buffer Waktu', line=dict(color='#5DADE2')))
                    fig_trend.add_hline(y=0, line_width=2, line_dash="dash", line_color="red")
                    
                    # --- FITUR BARU: Menambahkan penanda hari libur di grafik ---
                    holidays_in_chart = {date: name for date, name in holidays_data.items() if date in history_df['tanggal'].values}
                    for date, name in holidays_in_chart.items():
                        fig_trend.add_vline(x=date, line_width=1, line_dash="dot", line_color="orange", annotation_text=name, annotation_position="top left")
                        
                    fig_trend.update_layout(title_text="Pergerakan Buffer Waktu (Jam)", xaxis_title="Tanggal", yaxis_title="Buffer Waktu")
                    st.plotly_chart(fig_trend, use_container_width=True)
                else:
                    st.info("Tidak ada data historis untuk komponen ini.")
