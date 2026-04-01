import tensorflow as tf
import numpy as np
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from PIL import Image
import io

# ---------------- APP INIT ---------------- #
app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="."), name="static")

IMG_SIZE = 224

# ---------------- LOAD MODEL ---------------- #
# Load TensorFlow SavedModel (exported from Colab)
model = tf.saved_model.load("model/weednet_savedmodel")

# inference function
infer = model.signatures["serving_default"]

print("✅ SavedModel loaded successfully")


# ---------------- CLASS LABELS ---------------- #
class_names = [
    "Kena (Commplina benghalensio)",
    "Lavhala (Cyperus Rotundus)",
    "Gajar gavat (Parthenium hysterophorus)",
    "Graceful Sandmart (Euphorbia hypericifolia)",
    "Sicklepod (Senna obtusifolia)",
    "Harali (Cynodon dactylon)",
    "Dwarf cassia (Chamaecrista pumila)",
    "Punarnava (Boerhaavia diffusa)",
    "Lambs Quarter plant (Chenopodium)",
    "Little Mallow (Malva parviflora)",
    "Moti dudhi (Euphorbia geneculata)",
    "Obscure morning glory (Ipomoea obscura)",
    "Asian Pigeonwings (Clitoria ternatea)",
    "Bilayat (Argemone mexicana)",
    "Choti dudhi (Euphorbia hirta)",
    "Digitaria SP (Digitaria sanguinalis)"
]


# ---------------- HOME PAGE ---------------- #
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# ---------------- PREDICTION API ---------------- #
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()

    # read image
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))

    # preprocess
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    input_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)

    # inference
    output = infer(input_tensor)

    preds = list(output.values())[0].numpy()

    pred_index = int(np.argmax(preds))
    pred_class = class_names[pred_index]
    confidence = float(np.max(preds))

    return {
        "prediction": pred_class,
        "confidence": round(confidence * 100, 2)
    }
