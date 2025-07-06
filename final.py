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

# --- Kamus Koordinat Simulasi untuk Setiap Seksi & Gudang di Plant ---
# Menggunakan koordinat AHM Pegangsaan Dua yang akurat
seksi_coordinates = {
    "Warehouse AHM (Pegangsaan Dua)": {"lat": -6.1653, "lon": 106.9185},
    "Press": {"lat": -6.1648, "lon": 106.9180},
    "Welding": {"lat": -6.1650, "lon": 106.9188},
    "Painting Steel": {"lat": -6.1655, "lon": 106.9195},
    "Machining": {"lat": -6.1645, "lon": 106.9190},
    "Assy Engine": {"lat": -6.1642, "lon": 106.9198},
    "Gensub": {"lat": -6.1660, "lon": 106.9175},
    "Assy Unit": {"lat": -6.1638, "lon": 106.9200},
}


# --- Fungsi untuk Mengambil Data dari API ---
@st.cache_data(ttl=3600)
def get_public_holidays():
    """Mengambil data hari libur nasional dari API publik."""
    try:
        response = requests.get("https://api-harilibur.vercel.app/api")
        if response.status_code == 200:
            holidays = response.json()
            return {pd.to_datetime(h['holiday_date']): h['holiday_name'] for h in holidays if h.get('is_national_holiday')}
        return {}
    except requests.exceptions.RequestException:
        return {}

@st.cache_data(ttl=600)
def get_coords_from_nominatim(location_text):
    """Mengubah teks lokasi menjadi koordinat menggunakan Nominatim API."""
    headers = {'User-Agent': 'AHM-PPIC-Dashboard/1.0'}
    params = {'q': location_text, 'format': 'json', 'limit': 1, 'countrycodes': 'id'}
    try:
        response = requests.get('https://nominatim.openstreetmap.org/search', params=params, headers=headers)
        if response.status_code == 200:
            data = response.json()
            if data:
                return {'lon': float(data[0]['lon']), 'lat': float(data[0]['lat'])}, "Berhasil mendapatkan koordinat."
            else:
                return None, f"Lokasi '{location_text}' tidak ditemukan."
        else:
            return None, f"Error Geocoding: Status Code {response.status_code}"
    except Exception as e:
        return None, f"Gagal menghubungi API Geocoding: {e}"


@st.cache_data(ttl=600)
def get_eta_from_ors(api_key, start_coords, end_coords):
    """Mengambil estimasi waktu tempuh dari OpenRouteService API."""
    if not api_key: return None, "API Key OpenRouteService belum dimasukkan."
    if not start_coords: return None, "Koordinat asal tidak valid."

    headers = {'Authorization': api_key}
    params = {'start': f"{start_coords['lon']},{start_coords['lat']}", 'end': f"{end_coords['lon']},{end_coords['lat']}"}
    try:
        response = requests.get('https://api.openrouteservice.org/v2/directions/driving-car', params=params, headers=headers)
        if response.status_code == 200:
            data = response.json()
            duration_hours = data['features'][0]['properties']['summary']['duration'] / 3600
            route_coords = data['features'][0]['geometry']['coordinates']
            return duration_hours, route_coords
        else:
            return None, f"Error Rute: {response.json().get('error', 'Status Code ' + str(response.status_code))}"
    except Exception as e:
        return None, f"Gagal menghubungi API Rute: {e}"


# --- Fungsi Backend Lainnya ---
@st.cache_resource
def load_artifacts():
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

# --- Inisialisasi Session State ---
if 'adjusted_lead_time' not in st.session_state: st.session_state.adjusted_lead_time = None
if 'route_coords' not in st.session_state: st.session_state.route_coords = None
if 'start_location_name' not in st.session_state: st.session_state.start_location_name = ""

# --- Memuat Artefak dan Data API ---
model, konsumsi_map, df_raw, df_processed = load_artifacts()
holidays_data = get_public_holidays()

if model is None:
    st.error("File model/data tidak ditemukan! Jalankan notebook pelatihan terlebih dahulu.")
    st.stop()

# --- Tampilan Utama ---
st.title("🏭 Dashboard Pemantauan Material PPIC")
st.markdown("Sistem Peringatan Dini untuk Mencegah Potensi Downtime Lini Produksi.")

with st.expander("🗓️ Notifikasi Hari Libur Nasional Terdekat", expanded=False):
    today = datetime.now().date()
    next_7_days = [today + timedelta(days=i) for i in range(8)]
    upcoming_holidays = {date.strftime('%Y-%m-%d'): name for date, name in holidays_data.items() if date.date() in next_7_days}
    if upcoming_holidays:
        for date_str, name in upcoming_holidays.items():
            st.warning(f"**Perhatian:** Hari **{name}** akan berlangsung pada tanggal **{date_str}**.", icon="🗓️")
    else:
        st.success("Tidak ada hari libur nasional dalam 7 hari ke depan.", icon="✅")

tab1, tab2 = st.tabs(["📊 Dashboard Pemantauan Stok", "🚚 Simulasi & Analisis ETA"])

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
    if not data_on_date.empty:
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
        with col_data2:
            st.write("**Seksi Tujuan Paling Kritis (Merah & Kuning):**")
            if not filtered_dashboard.empty:
                hotspot_data = filtered_dashboard[filtered_dashboard['Status'].isin(['Merah', 'Kuning'])]
                if not hotspot_data.empty:
                    hotspot_count = hotspot_data.groupby('Seksi Tujuan')['Status'].count().sort_values(ascending=True)
                    fig_hotspot = go.Figure(go.Bar(y=hotspot_count.index, x=hotspot_count.values, orientation='h', marker_color='#C0392B'))
                    fig_hotspot.update_layout(height=500, title_text="Jumlah Item Kritis per Seksi", xaxis_title="Jumlah Item", yaxis_title="Seksi Tujuan")
                    st.plotly_chart(fig_hotspot, use_container_width=True)

with tab2:
    st.header("Analisis dan Simulasi ETA dari Supplier ke Gudang")
    
    with st.sidebar:
        st.title("⚙️ Analisis Detail & ETA")
        ors_api_key = st.text_input("Masukkan OpenRouteService API Key Anda", type="password", help="Kunci API diperlukan untuk menghitung rute dan ETA.")
        st.caption("Dapatkan kunci API gratis di [OpenRouteService Sign Up](https://openrouteservice.org/dev/#/signup)")
        
        st.markdown("---")
        st.subheader("1. Rute Pengiriman (Supplier ke Gudang)")
        start_location_text = st.text_input("Lokasi Asal Pengiriman (Supplier)", placeholder="Contoh: Bogor, Jawa Barat")
        st.text_input("Lokasi Tujuan di Plant", value="Warehouse AHM (Pegangsaan Dua)", disabled=True)
        
        # --- FITUR BARU: Faktor Penyesuaian Lalu Lintas ---
        traffic_factor = 100

        if st.button("Hitung Estimasi Lead Time (ETA)", use_container_width=True):
            with st.spinner("Mencari koordinat lokasi asal..."):
                start_coords, msg = get_coords_from_nominatim(start_location_text)
                if start_coords:
                    st.success(f"Lokasi '{start_location_text}' ditemukan.")
                    with st.spinner("Menghubungi API untuk menghitung ETA..."):
                        end_coords = seksi_coordinates["Warehouse AHM (Pegangsaan Dua)"]
                        eta_hours, route = get_eta_from_ors(ors_api_key, start_coords, end_coords)
                        
                        if eta_hours is not None:
                            # Terapkan faktor penyesuaian
                            adjusted_eta = eta_hours * (1 + traffic_factor / 100)
                            st.session_state.adjusted_lead_time = adjusted_eta
                            st.session_state.route_coords = route
                            st.session_state.start_location_name = start_location_text
                        else:
                            st.error(route) # Tampilkan pesan error dari API rute
                else:
                    st.error(msg)
        
        if st.session_state.adjusted_lead_time is not None:
            st.success(f"Estimasi Lead Time: **{st.session_state.adjusted_lead_time:.2f} jam**")
        
        st.markdown("---")
        st.subheader("2. Analisis Risiko Stok di Seksi")
        list_seksi_internal = [s for s in seksi_coordinates.keys() if "Warehouse" not in s]
        selected_seksi_produksi = st.selectbox("Pilih Seksi Tujuan Produksi", options=sorted(list_seksi_internal), key="sb_seksi_produksi")
        
        list_komponen = df_processed['nama_komponen'].unique()
        selected_komponen = st.selectbox("Pilih Nama Komponen", options=sorted(list_komponen), key="sb_komponen")
        input_stok = st.number_input("Simulasi Stok Tersedia di Seksi (unit)", min_value=0, value=100, step=10, key="ni_stok")
        predict_button = st.button("Jalankan Analisis Risiko", type="primary", use_container_width=True)

    if predict_button:
        if st.session_state.adjusted_lead_time is None:
            st.error("Harap hitung Estimasi Lead Time (ETA) terlebih dahulu sebelum menjalankan analisis.")
        else:
            konsumsi_per_jam = konsumsi_map.get((selected_seksi_produksi, selected_komponen))
            if konsumsi_per_jam is None:
                st.error(f"Tidak ada data konsumsi untuk **{selected_komponen}** di **{selected_seksi_produksi}**.")
            else:
                input_lead_time = st.session_state.adjusted_lead_time
                waktu_habis_stok = input_stok / konsumsi_per_jam if konsumsi_per_jam > 0 else float('inf')
                selisih_waktu = waktu_habis_stok - input_lead_time
                
                if selisih_waktu <= 0: final_prediksi = 'Merah'
                else:
                    input_data = pd.DataFrame([{'seksi_tujuan': selected_seksi_produksi, 'nama_komponen': selected_komponen,
                                                'stok_tersedia': input_stok, 'lead_time': input_lead_time}])
                    final_prediksi = model.predict(input_data)[0]
                    if final_prediksi == 'Merah': final_prediksi = 'Kuning'
                
                st.subheader(f"Hasil Analisis untuk: {selected_komponen} di Seksi {selected_seksi_produksi}")
                
                col_result1, col_result2 = st.columns(2)
                with col_result1:
                    st.markdown("##### Ringkasan Analisis")
                    if final_prediksi == 'Merah': st.error(f"**Status: MERAH (Bahaya)** 🚨", icon="🚨")
                    elif final_prediksi == 'Kuning': st.warning(f"**Status: KUNING (Waspada)** ⚠️", icon="⚠️")
                    else: st.success(f"**Status: HIJAU (Aman)** ✅", icon="✅")
                    
                    st.metric("Estimasi Lead Time (ETA)", f"{input_lead_time:.2f} Jam")
                    st.metric("Waktu Habis Stok", f"{waktu_habis_stok:.1f} Jam")
                    st.metric("Buffer Waktu", f"{selisih_waktu:.2f} Jam")
                
                with col_result2:
                    st.markdown("##### Visualisasi Risiko")
                    fig_bar = go.Figure(go.Bar(x=[waktu_habis_stok, input_lead_time], y=['Waktu Stok Habis', 'Estimasi Lead Time'],
                                               orientation='h', marker_color=['#3498DB', '#E74C3C'],
                                               text=[f"{waktu_habis_stok:.1f} jam", f"{input_lead_time:.2f} jam"], textposition='auto'))
                    fig_bar.update_layout(title_text='Perbandingan Waktu Habis Stok vs. ETA', xaxis_title="Waktu (jam)", showlegend=False, height=300, margin=dict(t=30, b=10))
                    st.plotly_chart(fig_bar, use_container_width=True)

                st.markdown("---")
                saran = f"RISIKO DOWNTIME! Stok akan habis sebelum material tiba. Segera hubungi supplier/ekspedisi untuk percepat pengiriman **{selected_komponen}**." if final_prediksi == 'Merah' else "Waspada! Stok berpotensi menipis. Koordinasikan dengan supplier/ekspedisi." if final_prediksi == 'Kuning' else "Stok material dalam kondisi aman. Lanjutkan pemantauan rutin."
                st.info(f"**Saran Tindakan:** {saran}", icon="💡")

                if st.session_state.route_coords:
                    st.subheader(f"Visualisasi Rute: {st.session_state.start_location_name} -> Warehouse AHM")
                    route_df = pd.DataFrame(st.session_state.route_coords, columns=['lon', 'lat'])
                    st.map(route_df)
