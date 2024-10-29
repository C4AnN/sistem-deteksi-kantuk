import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
import pygame
import time
import tempfile
from PIL import Image

# Inisialisasi pygame untuk memainkan suara
pygame.mixer.init()

# Fungsi untuk memainkan suara peringatan
def play_alert_sound():
    pygame.mixer.music.load('Alert.wav')  # Path ke file suara
    pygame.mixer.music.play()

# Inisialisasi model YOLO
model = YOLO("lastyolov8m.pt")  # Gantilah dengan path model YOLOv9 yang benar

# Streamlit pengaturan antarmuka
st.title("Sistem Deteksi Kantuk dengan YOLOv9")
st.write("Aplikasi ini menggunakan webcam, file video, atau gambar untuk mendeteksi kantuk dan memberikan peringatan suara.")

# Tempatkan frame video dan gambar dalam Streamlit
frame_window = st.image([])

# Variabel untuk mencegah suara berulang terlalu cepat
last_alert_time = 0
alert_interval = 5  # Interval dalam detik untuk menghindari peringatan berulang terlalu cepat

# Fungsi untuk melakukan deteksi kantuk pada video real-time dari webcam
def detect_drowsiness_from_webcam():
    global last_alert_time

    # Menggunakan webcam (ID 0)
    cap = cv2.VideoCapture(0)

    # Loop untuk real-time deteksi
    while True:
        ret, frame = cap.read()

        if not ret:
            st.error("Gagal mengambil frame dari webcam")
            break

        # Konversi frame dari BGR ke RGB agar sesuai dengan warna asli
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Jalankan deteksi pada frame dari webcam
        results = model(frame_rgb)

        # Ambil hasil pertama dan render deteksinya
        annotated_frame = results[0].plot()

        # Periksa jika ada deteksi objek dengan confidence > 0.85
        if len(results[0].boxes) > 0:
            for box in results[0].boxes:
                dconf = box.conf.item()  # Confidence dari deteksi
                dclass = box.cls.item()  # Kelas dari deteksi

                # Misal 1.0 adalah kelas orang mengantuk, dan confidence > 0.85
                if dconf > 0.85 and dclass == 1.0:
                    current_time = time.time()

                    # Cek jika sudah lewat interval waktu untuk memainkan suara
                    if current_time - last_alert_time > alert_interval:
                        play_alert_sound()
                        st.warning("Deteksi kantuk! Suara peringatan dimainkan.")
                        last_alert_time = current_time  # Update waktu terakhir suara dimainkan
                    break  # Keluar dari loop setelah mendeteksi kantuk

        # Tampilkan hasil deteksi di jendela Streamlit dengan warna yang sesuai
        frame_window.image(annotated_frame, channels="RGB")

        # Tambahkan penundaan agar tidak terlalu membebani CPU
        time.sleep(0.1)

    # Melepas resource
    cap.release()

# Fungsi untuk melakukan deteksi kantuk dari file video MP4
def detect_drowsiness_from_video(video_path):
    global last_alert_time

    # Membuka video menggunakan OpenCV
    cap = cv2.VideoCapture(video_path)

    # Loop untuk membaca video frame-by-frame
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            st.error("Gagal membaca frame dari video.")
            break

        # Konversi frame dari BGR ke RGB agar sesuai dengan warna asli
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Jalankan deteksi pada frame dari video
        results = model(frame_rgb)

        # Ambil hasil pertama dan render deteksinya
        annotated_frame = results[0].plot()

        # Periksa jika ada deteksi objek dengan confidence > 0.85
        if len(results[0].boxes) > 0:
            for box in results[0].boxes:
                dconf = box.conf.item()  # Confidence dari deteksi
                dclass = box.cls.item()  # Kelas dari deteksi

                # Misal 1.0 adalah kelas orang mengantuk, dan confidence > 0.85
                if dconf > 0.85 and dclass == 1.0:
                    current_time = time.time()

                    # Cek jika sudah lewat interval waktu untuk memainkan suara
                    if current_time - last_alert_time > alert_interval:
                        play_alert_sound()
                        st.warning("Deteksi kantuk! Suara peringatan dimainkan.")
                        last_alert_time = current_time  # Update waktu terakhir suara dimainkan
                    break  # Keluar dari loop setelah mendeteksi kantuk

        # Tampilkan hasil deteksi di jendela Streamlit dengan warna yang sesuai
        frame_window.image(annotated_frame, channels="RGB")

        # Tambahkan penundaan agar tidak terlalu membebani CPU
        time.sleep(0.1)

    # Melepas resource
    cap.release()

# Fungsi untuk melakukan deteksi kantuk dari gambar
def detect_drowsiness_from_image(image):
    # Konversi gambar ke format yang sesuai untuk model (RGB)
    image_rgb = np.array(image.convert("RGB"))

    # Jalankan deteksi pada gambar
    results = model(image_rgb)

    # Ambil hasil pertama dan render deteksinya
    annotated_image = results[0].plot()

    # Tampilkan hasil deteksi di jendela Streamlit dengan warna yang sesuai
    frame_window.image(annotated_image, channels="RGB")

    # Periksa jika ada deteksi objek dengan confidence > 0.85
    if len(results[0].boxes) > 0:
        for box in results[0].boxes:
            dconf = box.conf.item()  # Confidence dari deteksi
            dclass = box.cls.item()  # Kelas dari deteksi

            # Misal 1.0 adalah kelas orang mengantuk, dan confidence > 0.85
            if dconf > 0.85 and dclass == 1.0:
                play_alert_sound()
                st.warning("Deteksi kantuk pada gambar! Suara peringatan dimainkan.")

# Pilihan input: Webcam, Upload Video, atau Upload Gambar
input_option = st.selectbox("Pilih sumber input:", ("Webcam", "Upload Video MP4", "Upload Gambar"))

if input_option == "Webcam":
    # Menjalankan deteksi kantuk dari webcam
    if st.button('Mulai Deteksi Kantuk dari Webcam'):
        detect_drowsiness_from_webcam()

elif input_option == "Upload Video MP4":
    # Upload video file
    uploaded_video = st.file_uploader("Unggah file video MP4", type=["mp4"])

    if uploaded_video is not None:
        # Menyimpan video yang diunggah ke file temporer
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            temp_file.write(uploaded_video.read())
            temp_video_path = temp_file.name

        # Menjalankan deteksi kantuk dari video yang diunggah
        if st.button('Mulai Deteksi Kantuk dari Video'):
            detect_drowsiness_from_video(temp_video_path)

elif input_option == "Upload Gambar":
    # Upload gambar file
    uploaded_image = st.file_uploader("Unggah file gambar", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Membaca gambar yang diunggah
        image = Image.open(uploaded_image)

        # Menjalankan deteksi kantuk dari gambar yang diunggah
        if st.button('Mulai Deteksi Kantuk dari Gambar'):
            detect_drowsiness_from_image(image)