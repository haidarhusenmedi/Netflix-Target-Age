import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ==========================================
# 1. MEMUAT KEMBALI MODEL & ENCODER (DESERIALISASI)
# ==========================================
# Membuka file biner (.pkl) yang berisi matriks bobot dan aturan algoritma
try:
    with open('rf_age_model.pkl', 'rb') as file: model_rf = pickle.load(file)
    with open('scaler_age.pkl', 'rb') as file: scaler = pickle.load(file)
    with open('le_type.pkl', 'rb') as file: le_type = pickle.load(file)
    with open('le_genre.pkl', 'rb') as file: le_genre = pickle.load(file)
    with open('le_country.pkl', 'rb') as file: le_country = pickle.load(file)
except FileNotFoundError:
    st.error("Error: File .pkl tidak ditemukan! Pastikan semua file model sudah di-upload ke GitHub.")

# ==========================================
# 2. ANTARMUKA PENGGUNA (UI)
# ==========================================
st.title('🎬 Prediksi Target Usia Tayangan Netflix')
st.write('Aplikasi ini memproses parameter input menggunakan **Random Forest Classifier** untuk memprediksi probabilitas target usia tayangan.')

# [WAJIB] Mencantumkan metrik performa sesuai instruksi di lembar tugas
st.info('**Metrik Performa Model:** Akurasi = 84.86% | F1-Score = 90.22%')

st.markdown("---")
st.header('Masukkan 4 Parameter Tayangan:')

# Mengambil daftar kategori asli dari encoder agar dropdown website akurat
tipe_options = le_type.classes_
genre_options = le_genre.classes_
negara_options = le_country.classes_

# Membuat form input dengan layout 2 kolom
col1, col2 = st.columns(2)
with col1:
    input_year = st.number_input('1. Tahun Rilis', min_value=1900, max_value=2026, value=2020)
    input_type = st.selectbox('2. Tipe Tayangan', tipe_options)
    
with col2:
    input_genre = st.selectbox('3. Genre Utama', genre_options)
    input_country = st.selectbox('4. Negara Produksi', negara_options)

# ==========================================
# 3. PROSES KLASIFIKASI & MATEMATIKA PREDIKSI
# ==========================================
if st.button('Mulai Prediksi 🚀'):
    # A. Data Preprocessing: Mengubah teks inputan pengguna menjadi angka (Encoding)
    type_encoded = le_type.transform([input_type])[0]
    genre_encoded = le_genre.transform([input_genre])[0]
    country_encoded = le_country.transform([input_country])[0]
    
    # B. Menyusun Matriks Input (Urutannya WAJIB sama persis seperti saat training di Colab)
    # Urutan: ['year_numeric', 'type_encoded', 'genre_encoded', 'country_encoded']
    fitur_input = np.array([[input_year, type_encoded, genre_encoded, country_encoded]])
    
    # C. Standarisasi (Z-Score) agar seimbang dengan rentang perhitungan saat training
    fitur_input_scaled = scaler.transform(fitur_input)
    
    # D. Prediksi menggunakan pohon keputusan Random Forest
    hasil_prediksi = model_rf.predict(fitur_input_scaled)
    
    # E. Logika Penampilan Hasil
    st.markdown("---")
    st.subheader("💡 Hasil Klasifikasi (Output):")
    if hasil_prediksi[0] == 0:
        st.success("🟢 **Kategori: KIDS / TEENS** (Aman untuk Anak-anak & Remaja)")
        st.write("Model memprediksi tayangan ini tidak mengandung unsur dewasa berat berdasarkan korelasi genre, negara, dan tahun rilis.")
    else:
        st.error("🔴 **Kategori: ADULTS** (Khusus Dewasa 18+)")
        st.write("Model mendeteksi pola probabilitas tinggi bahwa tayangan ini ditujukan untuk pemirsa dewasa (TV-MA, R, NC-17).")