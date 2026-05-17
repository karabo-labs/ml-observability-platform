FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r app && useradd -r -g app -d /app -s /sbin/nologin app

# Copy and install requirements
COPY --chown=app:app app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=app:app app/ ./
COPY --chown=app:app assets/ ./assets/
COPY --chown=app:app data/ ./data/

USER app

EXPOSE 7860

CMD ["python", "app.py"]
