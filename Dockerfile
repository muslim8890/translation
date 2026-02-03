# --- STAGE 1: Build Frontend ---
FROM node:20-slim as frontend-build
WORKDIR /app/frontend

# Copy dependencies first for caching
COPY frontend/package*.json ./
RUN npm install

# Copy source and build
COPY frontend ./
RUN npm run build
RUN ls -R /app/frontend/dist


# --- STAGE 2: Build Backend & Serve ---
FROM python:3.10-slim

# 1. Install System Dependencies
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update --fix-missing && apt-get install -y --no-install-recommends \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 2. Set Working Directory
WORKDIR /app
ENV PYTHONPATH=/app/backend

# 3. Copy Python Requirements
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy Backend Code
COPY backend /app/backend

# 5. Copy Built Frontend from Stage 1
COPY --from=frontend-build /app/frontend/dist /app/frontend/dist

# 6. Expose Port
EXPOSE 8000

# 7. Run Command
# 7. Run Command
ENV DEPLOY_VERSION="v23-url-fix-final"
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
