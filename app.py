import cv2
from fastapi import FastAPI
import pickle
from datetime import datetime
from fastapi import BackgroundTasks, UploadFile, File

app = FastAPI()


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

    with open("models/trained_model.pt", "rb") as file:
        torch.load(file)

        if prediction:
            return "yes prediction"
        else:
            return {"hello world"}
