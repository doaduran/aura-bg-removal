FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# u2netp.onnx modelini (4MB) build zamanında indirip imaja gömüyoruz.
# rembg paketini artık kullanmadığımız için onun pooch ile runtime'da
# indirmesine gerek yok - hem ilk istek gecikmesiz başlar hem de imaj
# içinde model dosyası garanti olarak bulunur. curl işi bitince kaldırılıyor,
# imaj boyutu ve baseline bellek gereksiz yere büyümesin diye.
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && mkdir -p /app/models \
    && curl -L -o /app/models/u2netp.onnx \
       https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx \
    && echo "8e83ca70e441ab06c318d82300c84806  /app/models/u2netp.onnx" | md5sum -c - \
    && apt-get purge -y curl && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

COPY app.py .

# Farklı ücretsiz barındırma servisleri (Koyeb, Render, Cloud Run vb.) portu
# kendi PORT ortam değişkenleriyle veriyor, bu yüzden sabit değer yerine
# onu okuyoruz. Hiçbiri PORT vermezse 7860'a düşer (Hugging Face için).
EXPOSE 7860

CMD uvicorn app:app --host 0.0.0.0 --port ${PORT:-7860}
