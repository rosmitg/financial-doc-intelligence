FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt .

# Install Python dependencies.
# --no-deps: install the exact pinned closure from the working venv and skip
#   pip's resolver (the venv intentionally mixes langchain 0.3.25 with
#   langchain-core 1.6.2, which the resolver rejects but which works at runtime).
# torch CPU wheels (torch==2.14.0+cpu) live on the PyTorch index, not PyPI.
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --no-deps \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements.txt

# Copy application code
COPY . .

# Create storage directory
RUN mkdir -p storage

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "app.api.routes:app", "--host", "0.0.0.0", "--port", "8000"]