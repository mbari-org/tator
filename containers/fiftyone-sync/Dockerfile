# FiftyOne sync service for Tator. Build from repo root:
#   docker build -f containers/fiftyone-sync/Dockerfile -t fiftyone-sync .
FROM python:3.12-slim

WORKDIR /app

# Install system deps for Pillow / image handling
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    libjpeg-dev zlib1g-dev \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY services/fiftyone-sync/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN rm requirements.txt

COPY services/fiftyone-sync/ .

ENV PYTHONUNBUFFERED=1
EXPOSE 8001
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
