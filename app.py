import base64
import io
import os
from typing import Optional

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from PIL import Image
from rembg import remove, new_session

app = FastAPI()

# Model bir kere, sunucu ilk ayağa kalkarken yüklenir. Her istekte tekrar
# yüklenmediği için ikinci ve sonraki istekler çok daha hızlı olur.
session = new_session("u2net")

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
        input_image = Image.open(io.BytesIO(input_bytes)).convert("RGBA")
    except Exception:
        raise HTTPException(status_code=400, detail="Geçersiz görsel verisi")

    output_image = remove(input_image, session=session)
    output_image = trim_transparent_margin(output_image)

    buffer = io.BytesIO()
    output_image.save(buffer, format="PNG")
    result_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    # App.js / proxy'nin beklediği ile AYNI format: data.data.result_b64
    return {"data": {"result_b64": result_b64}}
