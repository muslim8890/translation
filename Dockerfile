# Base Image: Python 3.10 Slim (Lightweight)
# CACHE BUST: 2026-02-02-FINAL-FIX-V3
FROM python:3.10-slim

# 1. Install System Dependencies (ESSENTIAL for PDF/Poppler)
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update --fix-missing
RUN apt-get install -y --no-install-recommends \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 2. Set Working Directory
WORKDIR /app
ENV PYTHONPATH=/app/backend

# 3. Copy Requirements first (Docker Cache Layering)
COPY backend/requirements.txt .

# 4. Install Python Dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the Application Code
# We act as if we are in the root of the repo
COPY backend /app/backend
COPY frontend/dist /app/frontend/dist

# 6. Expose Port
EXPOSE 8000

# 7. Run Command
# Note: "backend.main:app" assumes we are at /app and backend is a package or folder
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
