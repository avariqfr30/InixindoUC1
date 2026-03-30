# VPS Simulation Deployment

Dokumen ini ditujukan untuk simulasi deployment tingkat awal di VPS, bukan production penuh.

## 1. Kebutuhan host
- Ubuntu 22.04 LTS atau setara
- Docker Engine dan Docker Compose
- Ollama terpasang di host VPS
- Model embedding tersedia di host Ollama

## 2. Persiapan host
Install Ollama di host, lalu pastikan daemon aktif.

Contoh langkah:
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama serve
ollama pull bge-m3:latest
```

Jika menggunakan model cloud, login Ollama di host sesuai akun yang dipakai.

## 3. Persiapan aplikasi
```bash
cp .env.example .env
mkdir -p generated app_assets
[ -f app_state.db ] || touch app_state.db
```

Sesuaikan `.env` jika perlu, terutama:
- `SERPER_API_KEY`
- `DEMO_MODE`
- `DATA_ACQUISITION_MODE`
- `OLLAMA_HOST`
- identitas perusahaan penyusun

## 4. Menjalankan aplikasi
```bash
docker compose up --build -d
```

Aplikasi akan terbuka di port `5500`.

## 5. Health check
```bash
curl http://SERVER_IP:5500/health
curl http://SERVER_IP:5500/ready
```

`/health` hanya mengecek proses aplikasi hidup.

`/ready` mengecek:
- koneksi ke Ollama
- ketersediaan basis data proyek
- status knowledge base
- akses ke state aplikasi lokal

## 6. Data yang harus dipersist
Volume lokal yang harus tetap ada:
- `projects.db`
- `db.csv`
- `app_state.db`
- `app_assets/`
- `generated/`

## 7. Catatan penting
- Queue saat ini masih in-memory, jadi simulasi VPS disarankan memakai satu instance app.
- Deployment ini cocok untuk uji internal dan simulasi beban ringan-menengah.
- Untuk shared multi-use-case yang lebih serius, langkah berikutnya adalah queue terpusat dan database operasional terpisah.
