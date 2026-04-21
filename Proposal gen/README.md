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
export APP_PROFILE=demo
export INTERNAL_DATA_SOURCE=demo
export INTERNAL_DATA_FALLBACK=none
export SERPER_API_KEY=isi_api_key
```

Untuk mode production:

```bash
export APP_PROFILE=production
export INTERNAL_DATA_SOURCE=api
export INTERNAL_DATA_FALLBACK=none
export FIRM_API_AUTH_MODE=basic
export FIRM_API_USERNAME=isi_username
export FIRM_API_PASSWORD=isi_password
export FIRM_API_URL=https://api.perusahaan-anda.com
export FIRM_API_CONFIG_FILE=/srv/apps/proposal-gen/internal_api_config.json
export SERPER_API_KEY=isi_api_key
```

Jika ingin mode production tetap bisa lanjut ketika API internal sedang bermasalah, aktifkan fallback demo secara eksplisit:

```bash
export APP_PROFILE=production
export INTERNAL_DATA_SOURCE=api
export INTERNAL_DATA_FALLBACK=demo
```

Contoh file ada di [internal_api_config.example.json](./internal_api_config.example.json). Dengan pendekatan ini, operator cukup mengganti satu file konfigurasi resource dan kredensial env tanpa perlu memahami mode `rest`, `dataset`, atau `generic`.

Jika operator tidak ingin mengedit `.env` manual, gunakan helper berikut:

```bash
python scripts/profilectl.py demo --env-file .env
python scripts/profilectl.py production --env-file .env --api-config /srv/apps/proposal-gen/internal_api_config.json
python scripts/profilectl.py production --env-file .env --api-config /srv/apps/proposal-gen/internal_api_config.json --fallback demo
```

### Deploy Sync dengan Filter Production vs In-House

Gunakan script berikut agar artefak test/example tidak ikut terkirim pada deploy production:

```bash
chmod +x scripts/deploy_sync.sh
```

Dry-run production (cek dulu file yang akan dikirim):

```bash
scripts/deploy_sync.sh --mode production --dry-run
```

Deploy production (sinkronisasi + restart service):

```bash
scripts/deploy_sync.sh --mode production
```

Deploy in-house/dev (tetap mengirim file example/test):

```bash
scripts/deploy_sync.sh --mode inhouse
```

Catatan:
- `production` mengecualikan file example/test seperti `.env.example`, `internal_api_config.example.json`, folder `tests/`, folder `examples/`, dan pola `test_*.py` / `*_test.py`.
- `production` juga menjalankan cleanup terarah di server untuk menghapus artifact test/example yang sudah terlanjur ada.
- `inhouse` tidak memakai filter tambahan tersebut.

### 4. Jalankan aplikasi

```bash
python -m main.app
```

Lalu buka:

```text
http://127.0.0.1:5000
```

## Integrasi Internal API yang Disarankan

Operator seharusnya hanya perlu memahami 2 status:

- `APP_PROFILE=demo`
- `APP_PROFILE=production`

Dan 2 knob internal data:

- `INTERNAL_DATA_SOURCE=demo|api`
- `INTERNAL_DATA_FALLBACK=none|demo`

Implementasi request API di bawahnya tetap mendukung pola lama (`rest`, `dataset`, `generic`) untuk kompatibilitas, tetapi itu sekarang dianggap detail implementasi, bukan hal yang wajib dipahami operator.

### Bentuk konfigurasi yang disarankan

Gunakan satu file resource manifest yang mendefinisikan:

- `request_defaults`
- `resources.firm_profile`
- `resources.project_standards`
- `resources.client_relationship`

Setiap resource cukup menjelaskan:

- request yang harus dipanggil
- bagian response yang dipakai
- filter record bila perlu
- apakah LLM fallback diizinkan

### Doctor command

Sebelum berpindah dari demo ke production, jalankan:

```bash
python -m main.doctor --format text
```

Atau versi JSON:

```bash
python -m main.doctor --format json
```

Command ini memeriksa:

- profile aktif
- internal data source/fallback
- file config API
- konektivitas Ollama
- kecukupan `firm_profile`
- kecukupan `project_standards`
- kecukupan `client_relationship`

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
