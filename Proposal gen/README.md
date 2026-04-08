# Proposal Generator

Aplikasi ini digunakan untuk membuat proposal konsultasi/proyek secara lebih cepat dalam format dokumen `.docx`.

Fokus utama aplikasi:
- membantu menyusun proposal berdasarkan input user
- menambahkan konteks dari data internal dan/atau OSINT
- menghasilkan dokumen proposal yang rapi dan siap diunduh

## Untuk Pengguna

### Cara Pakai Singkat

1. Buka aplikasi di browser.
2. Isi semua field utama pada form.
3. Jika perlu, jalankan analisis estimasi biaya.
4. Cek preview outline bila tersedia.
5. Klik generate proposal.
6. Tunggu proses selesai, lalu unduh file `.docx`.

### Urutan Penggunaan yang Disarankan

1. Pilih atau isi nama perusahaan klien.
2. Isi konteks organisasi atau tujuan inisiatif.
3. Isi permasalahan utama yang ingin diselesaikan.
4. Pilih klasifikasi kebutuhan.
5. Pilih jenis proposal.
6. Pilih jenis proyek.
7. Isi estimasi waktu.
8. Pilih atau isi framework/regulasi yang relevan.
9. Isi estimasi biaya, atau gunakan analisis biaya bila tersedia.
10. Klik generate untuk membuat proposal final.

### Penjelasan Field

| Field | Kegunaan |
| --- | --- |
| `Nama Perusahaan` | Nama klien yang akan dibuatkan proposal |
| `Konteks Organisasi` | Ringkasan tujuan, inisiatif, atau latar belakang proyek |
| `Permasalahan` | Masalah utama, pain points, atau kebutuhan klien |
| `Klasifikasi Kebutuhan` | Kategori kebutuhan seperti problem, opportunity, atau directive |
| `Jenis Proposal` | Jenis layanan atau engagement yang akan ditawarkan |
| `Jenis Proyek` | Bentuk proyek, misalnya strategic, implementation, transformation, dan lain-lain |
| `Estimasi Waktu` | Perkiraan durasi pelaksanaan proyek |
| `Potensi Framework` | Framework, regulasi, atau standar yang relevan |
| `Estimasi Biaya` | Nilai estimasi biaya proyek |

### Tentang Analisis Estimasi Biaya

Fitur analisis biaya hanya bisa digunakan setelah field yang diperlukan terisi lengkap.

Tujuannya:
- memberi gambaran awal rentang biaya
- menyesuaikan estimasi dengan durasi dan skala proyek
- membantu user sebelum generate proposal final

Catatan:
- hasil analisis biaya adalah bahan bantu, bukan keputusan komersial final
- pada mode tertentu, aplikasi dapat memakai aturan komersial internal sebagai dasar utama

### Hasil Akhir

Setelah proses berhasil, aplikasi akan mengunduh file proposal dalam format `.docx`.

Isi proposal mengikuti struktur standar aplikasi, termasuk:
- bab per bab proposal
- penyesuaian konteks klien
- model pembiayaan
- penutup dan informasi kontak firma penulis, jika tersedia dan terverifikasi

## Mode Aplikasi

### 1. Demo Mode

Mode ini cocok untuk uji coba, presentasi, atau pengembangan awal.

Perilaku utama:
- tetap bisa berjalan walaupun internal API belum lengkap
- beberapa data internal memakai fallback/demo data
- sebagian logika masih dapat memanfaatkan OSINT untuk membantu pengisian konteks

### 2. Staged Mode

Mode ini dipakai saat aplikasi mulai masuk tahap implementasi yang lebih serius.

Perilaku utama:
- data perusahaan penulis proposal lebih mengandalkan internal API
- histori hubungan dengan klien diambil dari internal API
- logika komersial dapat mengikuti aturan internal
- OSINT tetap dipakai untuk konteks publik klien, berita, industri, dan regulasi

## Sumber Data dalam Aplikasi

Secara umum, aplikasi memakai tiga sumber data:

- `Input user`
  Untuk kebutuhan proyek yang sedang diajukan

- `Data internal`
  Untuk data resmi perusahaan penulis proposal, standar delivery, hubungan klien, dan aturan komersial

- `OSINT`
  Untuk data publik seperti profil klien, berita terbaru, regulasi, dan tren industri

## Setup Singkat untuk Admin / Operator

### 1. Install dependency

```bash
pip install -r requirements.txt
```

### 2. Jalankan layanan model lokal bila dipakai

Aplikasi ini menggunakan Ollama sebagai backend model lokal.
Pastikan Ollama berjalan sebelum aplikasi dijalankan.

### 3. Atur environment variable bila diperlukan

Contoh paling umum:

```bash
export DEMO_MODE=true
export DATA_ACQUISITION_MODE=demo
export SERPER_API_KEY=isi_api_key
```

Untuk mode implementasi bertahap:

```bash
export DEMO_MODE=false
export DATA_ACQUISITION_MODE=staged
export FIRM_API_URL=https://api.perusahaan-anda.com/v1
export FIRM_API_INTEGRATION_MODE=rest
export FIRM_API_AUTH_MODE=bearer
export API_AUTH_TOKEN=isi_token_internal_api
export SERPER_API_KEY=isi_api_key
```

Jika internal API memakai Basic Auth:

```bash
export DEMO_MODE=false
export DATA_ACQUISITION_MODE=staged
export FIRM_API_URL=https://api.perusahaan-anda.com/v1
export FIRM_API_INTEGRATION_MODE=rest
export FIRM_API_AUTH_MODE=basic
export FIRM_API_USERNAME=isi_username
export FIRM_API_PASSWORD=isi_password
export SERPER_API_KEY=isi_api_key
```

Jika ingin integrasi internal API lebih mudah dirawat, gunakan satu file konfigurasi JSON:

```bash
export DEMO_MODE=false
export DATA_ACQUISITION_MODE=staged
export FIRM_API_URL=https://api.perusahaan-anda.com
export FIRM_API_INTEGRATION_MODE=generic
export FIRM_API_AUTH_MODE=basic
export FIRM_API_USERNAME=isi_username
export FIRM_API_PASSWORD=isi_password
export FIRM_API_CONFIG_FILE=/srv/apps/proposal-gen/internal_api_config.json
```

Contoh file ada di [internal_api_config.example.json](./internal_api_config.example.json). Dengan pendekatan ini, perubahan path, full URL endpoint, method, query param, body request, response path, atau mode dataset bisa dilakukan di satu file tanpa mengubah adapter Python.

### 4. Jalankan aplikasi

```bash
python -m main.app
```

Lalu buka:

```text
http://127.0.0.1:5000
```

## Integrasi Internal API yang Disarankan

Sekarang ada 2 mode integrasi:

- `FIRM_API_INTEGRATION_MODE=rest`
  Gunakan bila backend internal sudah punya endpoint terpisah per resource.
- `FIRM_API_INTEGRATION_MODE=dataset`
  Gunakan bila backend hanya punya satu endpoint dataset generik, misalnya `/api/Resource/dataset`.
- `FIRM_API_INTEGRATION_MODE=generic`
  Gunakan bila backend punya endpoint apa pun dengan shape JSON bebas, misalnya `https://xxx.com/api/tag`, dan Anda ingin aplikasi mengadaptasi payload itu ke workflow proposal.

Ketiganya memakai adapter yang sama di aplikasi, jadi logika proposal tidak perlu ikut diubah.

### Mode REST

Jika aplikasi akan dipakai di tahap lanjut, sebaiknya internal API menyiapkan data berikut:

- `/firm-profile`
- `/standards/{project_type}`
- `/client-relationship?client_name=...`

Tujuannya agar aplikasi dapat:
- menampilkan identitas firma penulis proposal dengan akurat
- menggunakan metodologi dan standar delivery internal
- membedakan klien baru dan klien existing berdasarkan data internal

### Mode Dataset

Jika backend saat ini baru menyediakan satu endpoint generik seperti:

- `POST /api/Resource/dataset`

maka aplikasi bisa diarahkan ke mode dataset. Konfigurasi JSON cukup menjelaskan:

- path endpoint dataset
- method request
- payload default seperti `{"dataset":"ReferenceDataset"}`
- lokasi array record di response
- nama field penanda resource
- filter untuk `project_type` dan `client_name`

Dengan begitu, saat struktur backend berubah, Anda cukup mengubah JSON config, bukan kelas adapter Python.

### Mode Generic

Mode ini paling fleksibel untuk integrasi internal database yang belum stabil atau masih berubah-ubah.

Anda cukup mendefinisikan per resource:

- endpoint penuh atau path relatif
- method request
- query param
- body JSON
- lokasi payload di response (`response_path`)
- filter record bila response berupa array

Jika struktur JSON tidak langsung cocok dengan field internal aplikasi, adapter akan:

1. mencoba membaca field memakai alias map bawaan,
2. lalu, bila masih lemah, memakai model untuk menafsirkan payload JSON mentah ke schema internal.

Artinya, endpoint seperti berikut bisa dipakai tanpa menulis adapter baru:

- `https://xxx.com/api/tag-firm-profile`
- `https://xxx.com/api/tag-standards`
- `https://xxx.com/api/tag-client-history`

Selama response berupa JSON dan konfigurasi `resources` diarahkan dengan benar, workflow proposal tetap bisa memakai data tersebut.

## Troubleshooting Singkat

### Aplikasi terbuka tetapi daftar perusahaan kosong

Kemungkinan:
- data `db.csv` / `projects.db` belum siap
- knowledge base belum ter-refresh

### Analisis biaya tidak bisa ditekan

Kemungkinan:
- masih ada field wajib yang belum diisi

### Proposal tidak jadi terunduh

Periksa:
- backend sedang berjalan
- Ollama aktif
- tidak ada error pada log terminal

### Informasi firma penulis belum lengkap

Pada Demo Mode, aplikasi bisa memakai data fallback.
Pada Staged Mode, data sebaiknya sudah tersedia dari internal API atau environment variable resmi.

## Ringkasan

Gunakan `Demo Mode` untuk uji coba cepat.
Gunakan `Staged Mode` saat internal API mulai siap dan data resmi perusahaan sudah bisa dihubungkan ke aplikasi.

Untuk pengguna akhir, yang terpenting adalah:
- isi form dengan lengkap
- gunakan analisis biaya bila diperlukan
- generate proposal
- unduh hasil `.docx`
