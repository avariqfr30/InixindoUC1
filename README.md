# Inixindo Enterprise Proposal Engine 📄🏢

Aplikasi ini adalah sistem *enterprise* berbasis web untuk mengotomatiskan penyusunan dokumen proposal strategis IT. Dibangun menggunakan arsitektur modular **Flask**, sistem ini memadukan kekuatan *Large Language Models* (LLM) via **Ollama**, *Retrieval-Augmented Generation* (RAG) via **ChromaDB**, dan mesin *Open Source Intelligence* (OSINT) untuk menyusun proposal yang sangat presisi, lengkap dengan visualisasi data otomatis.

Pada versi ini (V1.0), aplikasi telah mendukung pola arsitektur *Adapter* untuk integrasi API internal perusahaan dan telah di-*containerize* (Docker) untuk kemudahan *deployment* ke ekosistem *cloud* seperti AWS.

## ✨ Fitur Utama

* **Smart Discovery UI**: Antarmuka dinamis dengan *Intelligent Autocomplete* yang otomatis menyesuaikan saran *pain points* dan regulasi berdasarkan sektor industri klien.
* **Firm API Adapter**: Memiliki fitur `DEMO_MODE` untuk presentasi (*mock data*), yang dapat dimatikan (`False`) untuk beralih mengambil standar metodologi dan struktur tim secara langsung dari API internal perusahaan.
* **OSINT Engine**: Modul *researcher* yang otomatis mencari berita terbaru klien, mandat regulasi terkini, dan mengekstrak warna serta logo korporat (berbasis regex pintar) secara *real-time*.
* **Auto-Migration DB**: Tidak lagi bergantung murni pada *flat file*. Sistem akan otomatis mengonversi `db.csv` menjadi basis data relasional SQLite (`projects.db`) pada saat pertama kali dijalankan.
* **Production Ready**: Dilengkapi dengan konfigurasi Gunicorn dan Docker Compose untuk *deployment* tanpa hambatan.

## 📋 Prasyarat Sistem

* **Python 3.9+** (Jika menjalankan secara lokal tanpa Docker).
* **Ollama**: Menggunakan *Ollama Cloud Endpoint* atau berjalan sebagai *local daemon* di port `11434`.
* **Google Custom Search API**: Membutuhkan `API_KEY` dan `CX_ID` untuk mengaktifkan fitur riset OSINT dan ekstraksi visual.
* **Docker & Docker Compose** (Opsional, khusus untuk *deployment* di server *cloud*/AWS).

## 🚀 Instalasi Lokal (Development)

### 1. Persiapan Lingkungan Virtual
Sangat disarankan menggunakan *virtual environment* untuk mengisolasi dependensi aplikasi.

```bash
# Buat virtual environment
python3 -m venv venv

# Aktifkan virtual environment (Mac/Linux)
source venv/bin/activate
# ATAU untuk Windows
# venv\Scripts\activate
```

### 2. Instalasi Dependensi
Instal seluruh *library* yang dibutuhkan dengan perintah berikut:

```bash
pip install flask flask-cors pandas chromadb ollama matplotlib python-docx markdown beautifulsoup4 requests Pillow sqlalchemy gunicorn
```

### 3. Konfigurasi Sistem (`config.py`)
Buka file `config.py` dan perhatikan parameter kunci berikut sebelum menjalankan aplikasi:

* **Mode Operasi**: Set `DEMO_MODE = True` untuk pengujian lokal. Saat diserahkan ke tim IT, ubah menjadi `False` dan masukkan URL API perusahaan di variabel `FIRM_API_URL`.
* **Kredensial API**: Masukkan `GOOGLE_API_KEY` dan `GOOGLE_CX_ID`.
* **Routing AI**: Pastikan variabel `OLLAMA_HOST` mengarah ke *endpoint* yang tepat (lokal `http://127.0.0.1:11434` atau URL Ollama Cloud Anda).

### 4. Inisialisasi Basis Data
Aplikasi beroperasi menggunakan SQLite. Jika file `projects.db` belum ada, cukup letakkan file `db.csv` Anda di *root directory*. Pada saat *startup*, sistem akan otomatis membersihkan data, melakukan pemformatan (seperti format Rupiah), dan memigrasikannya secara permanen ke dalam SQLite.

### 5. Menjalankan Aplikasi
```bash
python app.py
```
Aplikasi dapat diakses melalui browser di `http://127.0.0.1:5000`.

---

## ☁️ Deployment ke Production (AWS / Cloud)

Untuk tahap serah terima (*handover*) ke tim infrastruktur/IT, aplikasi ini telah dikonfigurasi menggunakan Docker untuk metode *Lift and Shift* ke server seperti AWS EC2.

1.  Siapkan VM/Instance (contoh: AWS EC2 `t3.medium` jika komputasi AI sudah di-*offload* ke Ollama Cloud).
2.  Salin seluruh *source code* ke dalam server.
3.  Jalankan perintah berikut di terminal server:

```bash
docker-compose up -d --build
```

Arsitektur ini akan:
* Membangun *image* Python menggunakan server WSGI tangguh (Gunicorn) dengan 4 *worker*.
* Menjaga persistensi data SQLite dan *Vector Database* ChromaDB menggunakan *Docker Volumes*.
* Memetakan aplikasi langsung ke port 80 (HTTP) server Anda.

## 🛠️ Arsitektur Output Dokumen
Dokumen yang dihasilkan di-*render* secara *native* ke dalam format `.docx`. Grafik (Gantt, Bar, Flowchart) dihasilkan secara otomatis oleh *engine* `matplotlib` di *backend* menggunakan palet warna (HEX) yang disesuaikan dengan identitas visual klien hasil riset OSINT.