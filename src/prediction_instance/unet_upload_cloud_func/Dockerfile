FROM python:3.11.7-slim-bookworm

WORKDIR /app

COPY . /app

# Install GDAL dependencies
RUN apt-get update && apt-get install -y \
    libgdal-dev \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Set GDAL_CONFIG environment variable
ENV GDAL_CONFIG=/usr/bin/gdal-config

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8080

CMD ["gunicorn", "-b", ":8080", "main:app"]