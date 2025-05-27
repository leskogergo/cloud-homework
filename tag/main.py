import os
import redis
import cv2
import ast
from minio import Minio
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
        _, msg = redis_client.blpop("tag")
        name, object_string = msg.split("|", 1)
        objects = ast.literal_eval(object_string)

        response = minio_client.get_object(BUCKET, name)
        arr = bytearray(response.read())
        image = cv2.imdecode(np.asarray(arr, dtype=np.uint8), cv2.IMREAD_COLOR)

        tagged_image = draw_boxes(image, objects)
        _, result_bytes = cv2.imencode(".jpg", tagged_image)
        minio_client.put_object(
            BUCKET,
            "tagged/" + name,
            data=bytes(result_bytes),
            length=len(result_bytes),
            content_type="image/jpeg"
        )

thread = threading.Thread(target=process, daemon=True)
thread.start()

while True:
    time.sleep(1)
