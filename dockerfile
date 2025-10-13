# Gunakan base image Python 3.10 yang ramping (slim)
FROM python:3.10-slim

# Tetapkan direktori kerja di dalam container
WORKDIR /app

# Salin file requirements terlebih dahulu untuk optimasi cache Docker
COPY requirements.txt requirements.txt

# Instal Gunicorn (server WSGI production untuk Flask) dan dependensi lainnya
# --no-cache-dir digunakan agar ukuran image Docker tetap kecil
RUN pip install --no-cache-dir gunicorn
RUN pip install --no-cache-dir -r requirements.txt

# Salin seluruh kode aplikasi Anda (termasuk folder api dan models) ke dalam container
COPY . .

# Beritahu Docker bahwa aplikasi akan berjalan di port 5000
EXPOSE 5000

# Perintah untuk menjalankan server Gunicorn saat container dimulai
# Ini akan menjalankan objek 'app' dari file 'api/index.py'
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "api.index:app"]