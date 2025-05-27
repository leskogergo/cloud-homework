from fastapi import FastAPI
import os
import redis
from minio import Minio
import cv2
import numpy as np
import json
import time

app = FastAPI()

# MinIO és Redis kliens
minio_client = Minio(
    os.getenv("MINIO_ENDPOINT"),
    access_key=os.getenv("MINIO_ACCESS_KEY"),
    secret_key=os.getenv("MINIO_SECRET_KEY"),
    secure=False
)
redis_client = redis.Redis(host=os.getenv("REDIS_HOST"), port=6379, decode_responses=True)
BUCKET = "images"

def resize_image_job(message):
    data = json.loads(message)
    job_id = data["job_id"]
    path = data["image_path"]

    resp = minio_client.get_object(BUCKET, path)
    image_bytes = resp.read()
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    resized = cv2.resize(img, (img.shape[1] // 4, img.shape[0] // 4))
    new_path = f"{job_id}/resized.jpg"
    _, buffer = cv2.imencode(".jpg", resized)
    minio_client.put_object(
        BUCKET,
        new_path,
        data=buffer.tobytes(),
        length=len(buffer),
        content_type="image/jpeg"
    )

    redis_client.rpush("grayscale", json.dumps({
        "job_id": job_id,
        "image_path": new_path
    }))

@app.on_event("startup")
def consume():
    while True:
        msg = redis_client.blpop("resize", timeout=0)
        if msg:
            _, body = msg
            resize_image_job(body)
        time.sleep(0.1)
