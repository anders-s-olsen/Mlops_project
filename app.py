import pickle
from datetime import datetime

import cv2
from fastapi import BackgroundTasks, FastAPI, File, UploadFile
from google.cloud import storage
from predict_model import predict

app = FastAPI()
BUCKET_NAME = "model_checkpoints_group24"
MODEL_FILE = "trained_model.pt"

client = storage.Client()
bucket = client.get_bucket(BUCKET_NAME)
blob = bucket.get_blob(MODEL_FILE)
my_model = pickle.loads(blob.download_as_string())


@app.get("/")
def main():
    return "welcome to mnist predictor"


def add_to_database(pred: int):
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

    return predict(my_model, image)
