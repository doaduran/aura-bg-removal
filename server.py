from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from rembg import remove, new_session
from PIL import Image, ImageOps, ImageEnhance
import io
import base64
import os
import uvicorn

app = FastAPI(title="Aura AI Studio Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 512MB RAM limitine tam oturan hafif model (u2netp)
session = new_session("u2netp")

def enhance_cloth_image(img: Image.Image) -> Image.Image:
    r, g, b, a = img.split()
    rgb_img = Image.merge('RGB', (r, g, b))
    rgb_img = ImageOps.autocontrast(rgb_img, cutoff=1)
    
    enhancer_color = ImageEnhance.Color(rgb_img)
    rgb_img = enhancer_color.enhance(1.08)
    
    enhancer_sharp = ImageEnhance.Sharpness(rgb_img)
    rgb_img = enhancer_sharp.enhance(1.15)
    
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
        
        # Hafif segmentasyon
        segmented = remove(input_image, session=session)
        
        # Renk ve stüdyo canlılığı optimizasyonu
        final_image = enhance_cloth_image(segmented)
        
        buffered = io.BytesIO()
        final_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        return {"image_base64": f"data:image/png;base64,{img_str}"}
    except Exception as e:
        return {"error": str(e)}, 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
