import base64
import gc
import io
import os
from typing import Optional

# onnxruntime, "OMP_NUM_THREADS" ortam değişkenini kendi thread sayısı için
# okuyor (rembg'nin session_factory.py'si bunu böyle kullanıyor). Bunu 1'e
# sabitlemek, session import edilip oluşturulmadan ÖNCE ayarlanmalı.
os.environ.setdefault("OMP_NUM_THREADS", "1")

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from PIL import Image
from rembg import remove, new_session

app = FastAPI()

# u2netp, u2net'in hafifletilmiş versiyonu (4MB, u2net ise 176MB).
# Render'ın ücretsiz planındaki 512MB bellek sınırına sığmak için bunu kullanıyoruz.
session = new_session("u2netp")

# Telefon fotoğrafları genelde çok yüksek çözünürlüklü oluyor (örn. 4000x3000px),
# bu da işlerken gereksiz yere fazla bellek harcatıyor. İşlemeden önce
# uzun kenarı bu boyuta küçültüyoruz, kalite için yeterli, bellek için çok daha hafif.
MAX_DIMENSION = 1024

# İsteğe bağlı basit koruma: Hugging Face Space ayarlarından "PROXY_SECRET"
# adında bir secret eklersen, sadece doğru anahtarla gelen istekler kabul
# edilir. Eklemezsen bu kontrol devre dışı kalır, hiçbir şeyi bozmaz.
PROXY_SECRET = os.environ.get("PROXY_SECRET")


class RemoveBgRequest(BaseModel):
    image_file_b64: str


def trim_transparent_margin(img: Image.Image, margin_percent: float = 0.08) -> Image.Image:
    """Kıyafetin etrafındaki şeffaf boşluğu kırpar, ufak bir pay bırakarak."""
    alpha = img.split()[-1]
    bbox = alpha.getbbox()
    if bbox is None:
        return img
    left, top, right, bottom = bbox
    width, height = right - left, bottom - top
    margin_x = int(width * margin_percent)
    margin_y = int(height * margin_percent)
    left = max(0, left - margin_x)
    top = max(0, top - margin_y)
    right = min(img.width, right + margin_x)
    bottom = min(img.height, bottom + margin_y)
    return img.crop((left, top, right, bottom))


@app.get("/")
def health():
    # Hugging Face'in "Running" durumunu göstermesi ve senin tarayıcıdan
    # test edebilmen için basit bir sağlık kontrolü.
    return {"status": "ok", "service": "aura-bg-removal"}


@app.post("/removebg")
def remove_bg(payload: RemoveBgRequest, x_proxy_secret: Optional[str] = Header(default=None)):
    if PROXY_SECRET and x_proxy_secret != PROXY_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        input_bytes = base64.b64decode(payload.image_file_b64)
        input_image = Image.open(io.BytesIO(input_bytes))

        # ÖNEMLİ: JPEG'lerde draft(), dosyayı TAM ÇÖZÜNÜRLÜKTE decode etmeden
        # önce JPEG'in kendi katmanlı yapısını kullanarak küçük boyutta decode
        # etmeyi sağlar. Bu olmadan, thumbnail() çağrılsa bile telefon
        # fotoğrafı (örn. 4032x3024) önce TAM boyutuyla belleğe açılıyordu -
        # OOM'un asıl sebebi buydu. PNG'lerde draft() etkisizdir, zararı olmaz.
        input_image.draft("RGB", (MAX_DIMENSION, MAX_DIMENSION))
        input_image = input_image.convert("RGBA")
    except Exception:
        raise HTTPException(status_code=400, detail="Geçersiz görsel verisi")

    # Draft sonrası boyut hâlâ MAX_DIMENSION'ı aşabilir (draft sadece 1/2, 1/4, 1/8
    # gibi katlarda küçültür), bu yüzden thumbnail ile kesin sınıra çekiyoruz.
    input_image.thumbnail((MAX_DIMENSION, MAX_DIMENSION), Image.LANCZOS)

    output_image = remove(input_image, session=session)
    output_image = trim_transparent_margin(output_image)

    buffer = io.BytesIO()
    output_image.save(buffer, format="PNG")
    result_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    # İşlem biten büyük görselleri hemen bellekten temizle, sıradaki isteğe temiz başla
    del input_image, output_image, input_bytes, buffer
    gc.collect()

    # App.js / proxy'nin beklediği ile AYNI format: data.data.result_b64
    return {"data": {"result_b64": result_b64}}
