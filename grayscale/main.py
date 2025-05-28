import os
import redis
import json
from minio import Minio
import cv2
import numpy as np
from fastapi import FastAPI
import threading
import time

# Konfigurációs változók
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY")
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_QUEUE = os.getenv("REDIS_QUEUE", "grayscale")
NEXT_QUEUE = os.getenv("NEXT_QUEUE", "objectdetect")
BUCKET = "grayscale"

# Kliensek inicializálása
minio_client = Minio(
    MINIO_ENDPOINT,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=False
)

if not minio_client.bucket_exists(BUCKET):
    minio_client.make_bucket(BUCKET)

redis_client = redis.Redis(host=REDIS_HOST, port=6379, decode_responses=True)

app = FastAPI()

def process():
    while True:
        _, raw_msg = redis_client.blpop(REDIS_QUEUE)
        data = json.loads(raw_msg)
        job_id = data["job_id"]
        source_bucket = data["bucket"]
        image_path = data["image_path"]

        response = minio_client.get_object(source_bucket, image_path)
        img_array = np.asarray(bytearray(response.read()), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, encoded_image = cv2.imencode(".jpg", gray)

        new_path = f"{job_id}/grayscale.jpg"
        minio_client.put_object(
            BUCKET,
            new_path,
            data=encoded_image.tobytes(),
            length=len(encoded_image),
            content_type="image/jpeg"
        )

        redis_client.rpush(NEXT_QUEUE, json.dumps({
            "job_id": job_id,
            "bucket": BUCKET,
            "image_path": new_path
        }))

@app.on_event("startup")
def start_worker():
    threading.Thread(target=process, daemon=True).start()
