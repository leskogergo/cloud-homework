import os
import redis
import cv2
import numpy as np
from minio import Minio
import json
import threading

# Konfiguráció
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY")
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_QUEUE = os.getenv("REDIS_QUEUE", "objectdetect")
NEXT_QUEUE = os.getenv("NEXT_QUEUE", "tag")
BUCKET = "objectdetect"

# MinIO & Redis
minio_client = Minio(
    MINIO_ENDPOINT,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=False
)

if not minio_client.bucket_exists(BUCKET):
    minio_client.make_bucket(BUCKET)

redis_client = redis.Redis(host=REDIS_HOST, port=6379, decode_responses=True)

# Objektumdetekciós háló betöltése
net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt", "MobileNetSSD_deploy.caffemodel")
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle",
           "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
           "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

def process():
    while True:
        _, raw_msg = redis_client.blpop(REDIS_QUEUE)
        data = json.loads(raw_msg)
        job_id = data["job_id"]
        source_bucket = data["bucket"]
        image_path = data["image_path"]

        response = minio_client.get_object(source_bucket, image_path)
        image_array = np.asarray(bytearray(response.read()), dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)),
                                     0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()

        objects = []

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.2:
                idx = int(detections[0, 0, i, 1])
                label = CLASSES[idx]
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                objects.append({
                    "label": label,
                    "startX": startX,
                    "startY": startY,
                    "endX": endX,
                    "endY": endY
                })

        redis_client.rpush(NEXT_QUEUE, json.dumps({
            "job_id": job_id,
            "bucket": source_bucket,
            "image_path": image_path,
            "objects": objects
        }))

# Indítás FastAPI nélkül, csak worker
if __name__ == "__main__":
    process()
