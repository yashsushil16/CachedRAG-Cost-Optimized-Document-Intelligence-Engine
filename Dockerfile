# Stage 1: Build React Frontend
FROM node:20-alpine AS frontend-builder
WORKDIR /frontend
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ ./
# Use empty string so Vite calls backend endpoints relative to the same host (avoiding CORS issues)
ENV VITE_API_BASE=""
RUN npm run build

# Stage 2: Build Python Backend & Serve Frontend
FROM python:3.11-slim
WORKDIR /app

# Install python dependencies
COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend source
COPY backend/app ./app
COPY backend/test_api.py ./

# Create writable data directory for cache files
RUN mkdir -p /app/data && chmod -R 777 /app/data

# Copy built frontend static files
COPY --from=frontend-builder /frontend/dist ./static_frontend

# Expose port (Hugging Face Spaces runs on 7860 by default)
EXPOSE 7860

# Start server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
