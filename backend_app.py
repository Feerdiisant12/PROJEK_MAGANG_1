from flask import Flask, jsonify, request
from dotenv import load_dotenv
import gspread
import pandas as pd
import joblib
import os
import google.generativeai as genai
from datetime import datetime, timedelta
import numpy as np
import re

load_dotenv()

app = Flask(__name__)

# --- Fungsi Inisialisasi dan Koneksi ---
def initialize_gemini():
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set!")
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        print("✅ API Gemini berhasil dikonfigurasi.")
        return model
    except Exception as e:
        print(f"❌ Error konfigurasi Gemini: {e}")
        return None

def load_models():
    try:
        model_crit = joblib.load('criticality_model.pkl')
        model_anom = joblib.load('anomaly_model.pkl')
        scaler = joblib.load('scaler.pkl')
        print("✅ Model Machine Learning berhasil dimuat.")
        return model_crit, model_anom, scaler
    except Exception as e:
        print(f"❌ Error memuat model .pkl: {e}")
        return None, None, None

def connect_gspread():
    try:
        gc = gspread.service_account(filename='credentials.json')
        spreadsheet = gc.open("Monitoring part")
        ws_monitoring = spreadsheet.worksheet("template_monitoring")
        ws_rekap = spreadsheet.worksheet("REKAP") 
        ws_daily_meeting = spreadsheet.worksheet("Daily Meeting")
        print("✅ Koneksi Google Sheets berhasil, semua sheet ditemukan.")
        return ws_monitoring, ws_rekap, ws_daily_meeting
    except gspread.exceptions.WorksheetNotFound as e:
        print(f"❌ Error: Salah satu sheet tidak ditemukan di Google Sheets. Pastikan nama sheet sudah benar. Detail: {e}")
        return None, None, None
    except Exception as e:
        print(f"❌ Error koneksi Google Sheets: {e}")
        return None, None, None

gemini_model = initialize_gemini()
model_crit, model_anom, scaler = load_models()
ws_monitoring, ws_rekap, ws_daily_meeting = connect_gspread()


def get_clean_dataframe(worksheet):
    """
    Membaca data dari worksheet dengan cara yang lebih tangguh.
    """
    if not worksheet:
        return pd.DataFrame()
    
    all_values = worksheet.get_all_values()
    if not all_values or len(all_values) < 2:
        return pd.DataFrame()

    header = all_values[0]
    data = all_values[1:]
    
    df = pd.DataFrame(data, columns=header)
    df = df.loc[:, ~df.columns.isin([''])]
    df.replace('', np.nan, inplace=True)
    # Standarisasi nama kolom menjadi uppercase di sini agar konsisten
    df.columns = [str(col).upper().strip() for col in df.columns]
    return df


def process_monitoring_data(df):
    # Nama kolom sudah di-standardize di get_clean_dataframe
    required_sheet_columns = ['STOCK WH1', 'WIP', 'PLAN']
    if not all(col in df.columns for col in required_sheet_columns):
        raise ValueError(f"Kolom tidak ditemukan di sheet 'template_monitoring'. Pastikan ada: {required_sheet_columns}")

    df_predict = df[required_sheet_columns].copy()
    df_predict.rename(columns={
        'STOCK WH1': 'Stock Level',
        'WIP': 'WIP',
        'PLAN': 'Deviation'
    }, inplace=True)

    for col in df_predict.columns:
        df_predict[col] = pd.to_numeric(df_predict[col], errors='coerce')
    
    df_predict.dropna(inplace=True)
    return df_predict

# --- API Endpoints ---
@app.route('/api/monitoring_data', methods=['GET'])
def get_monitoring_data():
    if not all([model_crit, model_anom, scaler, ws_monitoring]):
        return jsonify({"error": "Sistem belum siap. Cek model, koneksi, atau API."}), 503
    df_raw = get_clean_dataframe(ws_monitoring)
    if df_raw.empty:
        return jsonify({"error": "Tidak ada data di sheet 'template_monitoring'."}), 404
    try:
        df_processed = process_monitoring_data(df_raw)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    if df_processed.empty:
        return jsonify({"error": "Data valid untuk prediksi tidak ditemukan."}), 404
    
    crit_prediction = model_crit.predict(df_processed)
    anomaly_prediction = model_anom.predict(df_processed[['Stock Level', 'WIP']])
    anomaly_label = ['Anomali' if x == -1 else 'Normal' for x in anomaly_prediction]

    df_raw.loc[df_processed.index, 'Prediksi_Kritis'] = crit_prediction
    df_raw.loc[df_processed.index, 'Status_Stok'] = anomaly_label
    
    df_raw['PLAN_NUMERIC'] = pd.to_numeric(df_raw['PLAN'], errors='coerce')
    df_raw['STOCK_WH1_NUMERIC'] = pd.to_numeric(df_raw['STOCK WH1'], errors='coerce')
    
    daily_consumption = df_raw['PLAN_NUMERIC'] / 22 
    df_raw['Days_of_Stock'] = (df_raw['STOCK_WH1_NUMERIC'] / daily_consumption).round(1)

    return jsonify(df_raw.to_dict(orient='records'))

@app.route('/api/morning_meeting_dashboard', methods=['GET'])
def get_morning_meeting_dashboard():
    if not ws_rekap:
        return jsonify({"error": "Koneksi ke sheet 'REKAP' gagal."}), 503
    
    df_rekap = get_clean_dataframe(ws_rekap)
    if df_rekap.empty:
        return jsonify({"error": "Tidak ada data di sheet 'REKAP'."}), 404
    
    required_original_cols = ['PART NUMBER', 'PART NAME', 'TYPE', 'ORDER + TOLERANSI - KEDATANGAN']
    if not all(col in df_rekap.columns for col in required_original_cols):
        missing_cols = [col for col in required_original_cols if col not in df_rekap.columns]
        return jsonify({"error": f"Kolom esensial tidak ditemukan di sheet 'REKAP'. Kolom yang hilang: {missing_cols}"}), 400

    df_rekap.rename(columns={
        'PART NUMBER': 'Part_Number',
        'PART NAME': 'Part_Name',
        'ORDER + TOLERANSI - KEDATANGAN': 'Stock_Health'
    }, inplace=True)
    
    df_rekap['Stock_Health'] = pd.to_numeric(df_rekap['Stock_Health'], errors='coerce').fillna(0)
    
    top_critical_parts = df_rekap.sort_values(by='Stock_Health', ascending=True).head(10)
    summary_by_type = df_rekap.groupby('TYPE')['Stock_Health'].agg(['mean', 'count', 'sum']).reset_index()
    summary_by_type.rename(columns={'mean': 'Avg_Stock_Health', 'count': 'Part_Count', 'sum': 'Total_Stock_Health'}, inplace=True)
    summary_by_type['Avg_Stock_Health'] = summary_by_type['Avg_Stock_Health'].round(2)
    summary_by_type = summary_by_type.sort_values(by='Avg_Stock_Health', ascending=True)

    total_parts = len(df_rekap)
    parts_in_deficit = len(df_rekap[df_rekap['Stock_Health'] < 0])
    parts_in_surplus = len(df_rekap[df_rekap['Stock_Health'] > 0])
    
    kpi_summary = {
        "total_parts_rekap": total_parts,
        "parts_deficit_count": parts_in_deficit,
        "parts_surplus_count": parts_in_surplus
    }

    dashboard_data = {
        "kpi_summary": kpi_summary,
        "top_critical_parts": top_critical_parts[['Part_Number', 'Part_Name', 'Stock_Health']].to_dict(orient='records'),
        "summary_by_type": summary_by_type.to_dict(orient='records')
    }
    return jsonify(dashboard_data)

# --- PERBAIKAN PADA PROMPT INI ---
@app.route('/api/generate_monitoring_insight', methods=['POST'])
def generate_monitoring_insight():
    if not gemini_model:
        return jsonify({"insight_text": "Error: Konfigurasi API Gemini gagal."}), 503
    
    part_data = request.json
    if not part_data:
        return jsonify({"error": "Data part yang dipilih tidak disertakan."}), 400

    # Menggunakan nama kolom yang sudah di-standardize (UPPERCASE)
    prompt = f"""
    Anda adalah seorang spesialis PPIC (Production Planning and Inventory Control) yang sangat berpengalaman. Analisis data komponen berikut secara mendalam.

    **Data Komponen yang Dipilih:**
    - Nama Part: {part_data.get('PART NUMBER / PART NAME')}
    - Stok Gudang (WH1): {part_data.get('STOCK WH1')} unit
    - Work In Progress (WIP): {part_data.get('WIP')} unit
    - Rencana Kebutuhan Bulanan (PLAN): {part_data.get('PLAN')} unit
    - **Prediksi Tingkat Kritis (dari model ML): {part_data.get('Prediksi_Kritis')}**
    - **Status Stok (dari model ML): {part_data.get('Status_Stok')}**

    **Tugas Anda:**
    1.  Berikan **analisis singkat** mengenai kondisi part ini. Apakah aman, perlu diawasi, atau dalam bahaya?
    2.  Jika prediksinya **'High'** atau **'Medium'**, jelaskan risikonya dan berikan **langkah-langkah mitigasi** yang harus segera diambil oleh tim gudang atau purchasing.
    3.  Jika statusnya **'Anomali'**, jelaskan apa arti anomali tersebut dalam konteks stok dan WIP, dan berikan **instruksi investigasi** yang jelas.
    4.  Berdasarkan semua data, berikan **satu saran perbaikan** untuk pengelolaan part ini ke depannya.
    5.  Gunakan format markdown dengan poin-poin dan emoji untuk kejelasan.
    """
    
    try:
        response = gemini_model.generate_content(prompt)
        insight_text = response.text
        return jsonify({"insight_text": insight_text})
    except Exception as e:
        return jsonify({"insight_text": f"Gagal menghubungi API Gemini: {e}"})


@app.route('/api/generate_holistic_insight', methods=['POST'])
def generate_holistic_insight():
    if not gemini_model:
        return jsonify({"insight_text": "Error: Konfigurasi API Gemini gagal."}), 503
    
    request_data = request.json
    rekap_analysis = request_data.get('rekap_analysis')
    
    if not rekap_analysis:
        return jsonify({"error": "Data analisis rekap tidak disertakan."}), 400

    prompt = f"""
    Anda adalah seorang Manajer Operasional senior yang sedang memimpin rapat pagi.
    Berdasarkan rangkuman data kesehatan stok komponen berikut, berikan insight strategis dan rekomendasi yang tajam.

    RANGKUMAN DATA KESEHATAN STOK HARI INI:
    - Total Komponen Dimonitor: {rekap_analysis.get('kpi_summary', {}).get('total_parts_rekap')}
    - Jumlah Komponen Defisit (Stok < Kebutuhan): {rekap_analysis.get('kpi_summary', {}).get('parts_deficit_count')}
    - Top 5 Komponen Paling Kritis (Defisit Terbesar): {rekap_analysis.get('top_critical_parts', [])[:5]}
    - Rangkuman per Tipe Komponen (Rata-rata Kesehatan Stok): {rekap_analysis.get('summary_by_type')}

    TUGAS ANDA:
    1.  Mulai dengan kesimpulan umum tentang kondisi kesehatan stok komponen secara keseluruhan. Apa risiko terbesar saat ini?
    2.  Fokus pada 2-3 komponen paling kritis. Berikan instruksi yang sangat spesifik dan darurat kepada tim PPIC untuk setiap komponen tersebut.
    3.  Lihat rangkuman per tipe komponen. Apakah ada tipe komponen tertentu yang secara konsisten bermasalah? Berikan rekomendasi untuk investigasi lebih lanjut pada tipe tersebut.
    4.  Berikan satu rekomendasi strategis jangka panjang untuk mengurangi jumlah komponen yang defisit di masa depan.
    5.  Gunakan bahasa yang tegas, profesional, dan to-the-point. Format jawabanmu menggunakan markdown dengan emoji untuk penekanan.
    """
    
    try:
        response = gemini_model.generate_content(prompt)
        insight_text = response.text
        return jsonify({"insight_text": insight_text})
    except Exception as e:
        return jsonify({"insight_text": f"Gagal menghubungi API Gemini: {e}"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
