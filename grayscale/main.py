import os
import redis
from minio import Minio
import cv2
import numpy as np
from fastapi import FastAPI
import threading
import time

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY")
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
BUCKET = "images"

minio_client = Minio(
    MINIO_ENDPOINT,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=False
)

redis_client = redis.Redis(host=REDIS_HOST, port=6379, decode_responses=True)

app = FastAPI()

def process():
    while True:
        _, image_name = redis_client.blpop("grayscale")
        response = minio_client.get_object(BUCKET, image_name)
        img_array = np.asarray(bytearray(response.read()), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, encoded_image = cv2.imencode(".jpg", gray)
        new_name = f"gray-{image_name}"
        minio_client.put_object(
            BUCKET, new_name,
            data=encoded_image.tobytes(),
            length=len(encoded_image),
            content_type="image/jpeg"
        )
        redis_client.rpush("objectdetect", new_name)

@app.on_event("startup")
def start_worker():
    threading.Thread(target=process, daemon=True).start()
