import os
import redis
import cv2
import json
import numpy as np
from minio import Minio

# Konfiguráció
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY")
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_QUEUE = os.getenv("REDIS_QUEUE", "tag")
BUCKET = "tag"

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

def draw_boxes(image, objects):
    for obj in objects:
        label = obj["label"]
        startX, startY = obj["startX"], obj["startY"]
        endX, endY = obj["endX"], obj["endY"]
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
        cv2.putText(image, label, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image

def process():
    while True:
        _, msg = redis_client.blpop(REDIS_QUEUE)
        data = json.loads(msg)
        job_id = data["job_id"]
        source_bucket = data["bucket"]
        image_path = data["image_path"]
        objects = data["objects"]

        response = minio_client.get_object(source_bucket, image_path)
        arr = bytearray(response.read())
        image = cv2.imdecode(np.asarray(arr, dtype=np.uint8), cv2.IMREAD_COLOR)

        tagged_image = draw_boxes(image, objects)
        _, result_bytes = cv2.imencode(".jpg", tagged_image)

        output_path = f"{job_id}/tagged.jpg"
        minio_client.put_object(
            BUCKET,
            output_path,
            data=bytes(result_bytes),
            length=len(result_bytes),
            content_type="image/jpeg"
        )

if __name__ == "__main__":
    process()
