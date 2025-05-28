# Alap image
FROM python:3.10

# Rendszerfüggőségek telepítése OpenCV-hez
RUN apt-get update && \
    apt-get install -y libgl1 && \
    rm -rf /var/lib/apt/lists/*

# Munkakönyvtár beállítása
WORKDIR /app

# Függőségek telepítése
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Összes fájl bemásolása, beleértve a modellfájlokat is
COPY . .

# Futási parancs
CMD ["python", "main.py"]
