from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from rembg import remove, new_session
from PIL import Image, ImageOps, ImageEnhance
import io
import base64
import uvicorn

app = FastAPI(title="Aura AI Pro Studio Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Yüksek hassasiyetli e-ticaret/kıyafet kesim modeli (isnet-general-use)
print("⏳ AI Segmentasyon Modeli Yükleniyor...")
session = new_session("isnet-general-use")
print("✅ AI Modeli Hazır!")

def enhance_cloth_image(img: Image.Image) -> Image.Image:
    """Kıyafetin ışık, gölge ve renk canlılığını stüdyo moduna getirir."""
    # RGBA kanallarına ayır
    r, g, b, a = img.split()
    rgb_img = Image.merge('RGB', (r, g, b))
    
    # 1. Otomatik Pozlama & Kontrast Dengeleme (Sarı/karanlık ışığı temizler)
    rgb_img = ImageOps.autocontrast(rgb_img, cutoff=1)
    
    # 2. Renk Canlılığını ve Netliği hafif optimize et
    enhancer_color = ImageEnhance.Color(rgb_img)
    rgb_img = enhancer_color.enhance(1.08)
    
    enhancer_sharp = ImageEnhance.Sharpness(rgb_img)
    rgb_img = enhancer_sharp.enhance(1.15)
    
    # Şeffaf Alpha kanalıyla tekrar birleştir
    r2, g2, b2 = rgb_img.split()
    return Image.merge('RGBA', (r2, g2, b2, a))

@app.get("/")
def read_root():
    return {"status": "Aura AI Studio Online 🤍"}

@app.post("/remove-bg")
async def remove_background(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        input_image = Image.open(io.BytesIO(contents)).convert("RGBA")
        
        # 1. Yüksek hassasiyetli arka plan kesimi (ISNet + Post Processing)
        segmented = remove(
            input_image, 
            session=session,
            alpha_matting=True,
            alpha_matting_foreground_threshold=240,
            alpha_matting_background_threshold=10,
            post_process_mask=True
        )
        
        # 2. Işık ve kumaş rengi optimizasyonu
        final_image = enhance_cloth_image(segmented)
        
        # 3. Base64 olarak geri döndür
        buffered = io.BytesIO()
        final_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        return {"image_base64": f"data:image/png;base64,{img_str}"}
    except Exception as e:
        print(f"Hata oluştu: {e}")
        return {"error": str(e)}, 500

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)