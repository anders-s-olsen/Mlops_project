from fastapi import FastAPI
import sys
from fastapi import UploadFile, File
from google.cloud import storage
sys.path.append('../src/models')
from predict_model import predict
import os.path

app = FastAPI()
BUCKET_NAME = "model_checkpoints_group24"
MODEL_FILE = "trained_model.pt"
storage_client = storage.Client()
bucket = storage_client.bucket(BUCKET_NAME)
blob = bucket.blob(MODEL_FILE)
if  not os.path.isfile("model.pt"):
    blob.download_to_filename("model.pt")

@app.get("/")
def main():
    return "welcome to mnist predictor"


def add_to_database(pred: str):
    with open("prediction_database.csv", "a") as file:
        file.write(str(pred))


@app.get("/test/")
def test(number: int):
    return number**2


@app.post("/predict/")
async def predict_number(input: UploadFile = File(...)):
    with open("image.jpg", "wb") as image:
        content = await input.read()
        image.write(content)
        image.close()
    
    pred = predict(img='image.jpg',model_state="model.pt")

    add_to_database(pred)
    return pred
#