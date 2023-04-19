from fastapi import FastAPI, File
from utils import utils
import uvicorn
from utils.model import __version__

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Hello, FastAPI!!!", "model_version": __version__}

@app.post("/model")
def upload_file(file: bytes = File(...)): 
    img = utils.read_image(file)
    predictions = utils.predict_(img)
    return predictions

if __name__ == "__main__":
    uvicorn.run(app,host="127.0.0.1", port=8080)
