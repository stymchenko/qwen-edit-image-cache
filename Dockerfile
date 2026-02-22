FROM nvcr.io/nvidia/pytorch:25.02-py3

WORKDIR /app

# Додаємо встановлення системних бібліотек для роботи з зображеннями
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# flash-attn треба встановлювати окремо після torch, з no-build-isolation
RUN pip install flash-attn==2.7.4.post1 --no-build-isolation --no-cache-dir

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY handler.py .

CMD ["python", "-u", "handler.py"]
