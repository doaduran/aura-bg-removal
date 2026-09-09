import base64
import gc
import io
import os
from typing import Optional

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from PIL import Image

app = FastAPI()

# --- Model ---
# rembg paketini artık kullanmıyoruz. rembg'yi kurmak scipy, scikit-image,
# opencv-python-headless, pymatting gibi ağır kütüphaneleri de zorunlu
# olarak getiriyordu; bunlar sunucu hiç istek almadan sadece açılırken bile
# ~300-400MB taban bellek tüketimine sebep oluyordu (512MB sınırında bu,
# neredeyse tüm bütçeyi tek başına tüketiyordu). Bunun yerine u2netp.onnx
# modelini (4MB) doğrudan onnxruntime ile çalıştırıyoruz - Dockerfile bu
# modeli build sırasında indirip imaja gömüyor, burada sadece yüklüyoruz.
MODEL_PATH = os.environ.get("MODEL_PATH", "/app/models/u2netp.onnx")

sess_opts = ort.SessionOptions()
# Thread başına ayrılan ekstra bellek tamponunu sınırlamak için thread
# sayısını 1'e sabitliyoruz (eskiden OMP_NUM_THREADS ortam değişkenine
# güveniyorduk, artık SessionOptions üzerinden doğrudan ve garantili
# şekilde ayarlıyoruz).
sess_opts.intra_op_num_threads = 1
sess_opts.inter_op_num_threads = 1

session = ort.InferenceSession(
    MODEL_PATH, sess_options=sess_opts, providers=["CPUExecutionProvider"]
)
INPUT_NAME = session.get_inputs()[0].name

# u2netp'nin beklediği giriş boyutu ve ImageNet normalizasyon değerleri.
# rembg'nin session_base.py / sessions/u2netp.py kaynağıyla doğrulandı.
MODEL_INPUT_SIZE = (320, 320)
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

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


def predict_mask(img: Image.Image) -> Image.Image:
    """u2netp ile ön plan maskesi üretir. rembg'nin BaseSession.normalize()
    ve U2netpSession.predict() adımlarının birebir aynısı."""
    im = img.convert("RGB").resize(MODEL_INPUT_SIZE, Image.LANCZOS)

    im_ary = np.array(im).astype(np.float32)
    im_ary = im_ary / np.max(im_ary)

    tmp_img = np.zeros((im_ary.shape[0], im_ary.shape[1], 3), dtype=np.float32)
    tmp_img[:, :, 0] = (im_ary[:, :, 0] - MEAN[0]) / STD[0]
    tmp_img[:, :, 1] = (im_ary[:, :, 1] - MEAN[1]) / STD[1]
    tmp_img[:, :, 2] = (im_ary[:, :, 2] - MEAN[2]) / STD[2]
    tmp_img = tmp_img.transpose((2, 0, 1))
    ort_input = {INPUT_NAME: np.expand_dims(tmp_img, 0).astype(np.float32)}

    ort_outs = session.run(None, ort_input)
    pred = ort_outs[0][:, 0, :, :]
    ma, mi = np.max(pred), np.min(pred)
    pred = (pred - mi) / (ma - mi)
    pred = np.squeeze(pred)

    mask = Image.fromarray((pred * 255).astype(np.uint8), mode="L")
    mask = mask.resize(img.size, Image.LANCZOS)
    return mask


def remove_background(img: Image.Image) -> Image.Image:
    """rembg'nin naive_cutout()'u ile aynı: maskeyi alfa kanalı olarak kullanıp
    orijinal görseli şeffaf bir tuval üzerine composite eder."""
    mask = predict_mask(img)
    empty = Image.new("RGBA", img.size, 0)
    cutout = Image.composite(img.convert("RGBA"), empty, mask)
    return cutout


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

    output_image = remove_background(input_image)
    output_image = trim_transparent_margin(output_image)

    buffer = io.BytesIO()
    output_image.save(buffer, format="PNG")
    result_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    # İşlem biten büyük görselleri hemen bellekten temizle, sıradaki isteğe temiz başla
    del input_image, output_image, input_bytes, buffer
    gc.collect()

    # App.js / proxy'nin beklediği ile AYNI format: data.data.result_b64
    return {"data": {"result_b64": result_b64}}
