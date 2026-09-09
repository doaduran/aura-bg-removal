FROM python:3.11-slim

WORKDIR /app

# rembg'nin görsel işleme için ihtiyaç duyduğu sistem kütüphanesi
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

# Farklı ücretsiz barındırma servisleri (Koyeb, Render, Cloud Run vb.) portu
# kendi PORT ortam değişkenleriyle veriyor, bu yüzden sabit değer yerine
# onu okuyoruz. Hiçbiri PORT vermezse 7860'a düşer (Hugging Face için).
EXPOSE 7860

CMD uvicorn app:app --host 0.0.0.0 --port ${PORT:-7860}
