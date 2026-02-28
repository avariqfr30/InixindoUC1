# Inixindo Use Case 1
# AI Proposal Generator 📄🤖

Aplikasi ini adalah sistem pembuat dokumen proposal strategis otomatis berbasis web. Dibangun menggunakan **Flask**, aplikasi ini memanfaatkan model bahasa lokal (LLM) melalui **Ollama** dan sistem *Retrieval-Augmented Generation* (RAG) menggunakan **ChromaDB**. Aplikasi ini juga diperkaya dengan fitur pencarian web *real-time* untuk melengkapi profil klien secara otomatis.

## 📋 Prasyarat Sistem

Sebelum menjalankan aplikasi, pastikan sistem Anda memiliki komponen berikut:

1. **Python 3.9 atau lebih baru**: Terinstal dan dapat diakses melalui terminal.
2. **Ollama**: Aplikasi pihak ketiga untuk menjalankan model AI secara lokal pada `http://127.0.0.1:11434`.
3. **Google API Key & Custom Search Engine ID (CX)**: Untuk fitur ekstraksi data dan logo dari internet.
4. **File Database (`db.csv`)**: File berisi histori atau data proyek (*Knowledge Base*).

## 🚀 Langkah-langkah Instalasi

### 1. Persiapan Direktori dan Virtual Environment
Sangat disarankan untuk menggunakan *virtual environment* agar dependensi aplikasi ini tidak mengganggu proyek Python Anda yang lain.

Buka terminal dan jalankan perintah berikut:

bash
# Buat virtual environment
python3 -m venv venv

# Aktifkan virtual environment (Mac/Linux)
source venv/bin/activate
# ATAU untuk Windows
# venv\Scripts\activate


### 2. Instalasi Dependensi (Library)
Aplikasi ini membutuhkan beberapa pustaka eksternal. Instal semuanya menggunakan `pip`:

bash
pip install flask flask-cors pandas chromadb ollama matplotlib python-docx markdown beautifulsoup4 requests Pillow


### 3. Konfigurasi Sistem (`config.py`)
Buka file `config.py` dan sesuaikan beberapa variabel penting berikut:

* **Kredensial Google**: 
  Masukkan `GOOGLE_API_KEY` dan `GOOGLE_CX_ID` milik Anda. Jika ini tidak diisi, fitur pencarian web dan pengambilan logo akan dilewati.
* **Model AI**:
  Secara default, kode menggunakan `LLM_MODEL = "gpt-oss:120b-cloud"` dan `EMBED_MODEL = "bge-m3:latest"`. Sesuaikan model LLM dengan kapasitas spesifikasi komputer keras Anda (misalnya ubah ke `llama3:8b` atau `phi3:mini` jika mengalami kendala memori yang terbatas).

### 4. Menyiapkan Ollama (Lokal AI)
Pastikan aplikasi Ollama sudah berjalan di latar belakang. Buka terminal baru dan jalankan perintah berikut untuk mengunduh model yang dibutuhkan oleh sistem:

bash
# Mengunduh model embedding untuk ChromaDB
ollama pull bge-m3:latest

# Mengunduh model LLM utama (sesuaikan dengan yang Anda tulis di config.py)
ollama pull gpt-oss:120b-cloud


### 5. Menyiapkan File Database
Pastikan Anda menempatkan file bernama `db.csv` di folder yang sama dengan `app.py`. File ini setidaknya harus memiliki dua kolom (contoh di kode: entitas utama dan sub-entitas) agar dapat dimuat oleh `KnowledgeBase`.

## ▶️ Cara Menjalankan Aplikasi

Setelah semua langkah di atas selesai, Anda siap untuk menyalakan server lokal Flask:

1. Pastikan Anda masih berada di dalam *virtual environment*.
2. Jalankan aplikasi menggunakan Python:
   bash
   python app.py
   
3. Terminal akan menampilkan log bahwa server berjalan di `port=5000` (biasanya di `http://127.0.0.1:5000`).
4. Buka *web browser* Anda dan kunjungi alamat tersebut untuk memuat halaman `index.html`.
5. Masukkan topik dan sub-topik proposal, lalu klik tombol *Generate*. Sistem akan memproses data dan otomatis mengunduh file berekstensi `.docx` ke perangkat Anda dengan tipe MIME `application/vnd.openxmlformats-officedocument.wordprocessingml.document`.

## 🛠️ Catatan Tambahan
* Output *chart* dan visual (seperti Gantt chart, bar chart, dan flowchart) dihasilkan menggunakan `matplotlib` secara *in-memory* dan akan disisipkan otomatis ke dalam dokumen Word.
* Aplikasi dikonfigurasi dengan CORS yang diaktifkan melalui `Flask-CORS`.
