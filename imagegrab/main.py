from fastapi import FastAPI, UploadFile, File
import uuid
import redis
import json
import os
from minio import Minio
import io

app = FastAPI()

# Redis & MinIO konfiguráció
REDIS_QUEUE = os.getenv("REDIS_QUEUE", "resize")
redis_client = redis.Redis(host=os.getenv("REDIS_HOST"), port=6379, decode_responses=True)

minio_client = Minio(
    os.getenv("MINIO_ENDPOINT"),
    access_key=os.getenv("MINIO_ACCESS_KEY"),
    secret_key=os.getenv("MINIO_SECRET_KEY"),
    secure=False
)

BUCKET = "imagegrab"
if not minio_client.bucket_exists(BUCKET):
    minio_client.make_bucket(BUCKET)

@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    job_id = str(uuid.uuid4())
    contents = await file.read()
    image_path = f"{job_id}/original.jpg"

    # Feltöltés MinIO-ba
    minio_client.put_object(
        BUCKET,
        image_path,
        io.BytesIO(contents),
        length=len(contents),
        content_type="image/jpeg"
    )

    # Üzenet küldése Redis-be
    message = {
        "job_id": job_id,
        "bucket": BUCKET,
        "image_path": image_path
    }

    redis_client.rpush(REDIS_QUEUE, json.dumps(message))
    return {"job_id": job_id, "image_path": image_path}
