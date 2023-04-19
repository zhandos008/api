from io import BytesIO
from PIL import Image
from pathlib import Path
import numpy as np
from utils.model import *
from utils.model import __version__

BASE_DIR = Path(__file__).resolve(strict=True).parent

def read_image(encoded_img):
    decoded_img = Image.open(BytesIO(encoded_img)).convert('RGB') 
    return np.array(decoded_img)

def load_model(path):
    model = PiepleinePredictor(
        ocr_model_path=path,
        ocr_config=config
    )
    return model

def predict_(img):
    ocr_model_path = f"{BASE_DIR}/ocr_model_{__version__}.ckpt"
    model = load_model(ocr_model_path)
    pred_data = model(img)
    predictions = []
    for pred in pred_data["predictions"]:
        box = [int(point) for point in pred["box"]]
        predictions.append((box, pred["text"]))
    return predictions





