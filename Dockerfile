# Base image with CUDA support
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04 as base

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Development image
FROM base as development
RUN pip3 install --no-cache-dir -r requirements-dev.txt
ENV PYTHONPATH=/app
ENV CORTEX_ENV=development

# Production image
FROM base as production

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m cortex && \
    chown -R cortex:cortex /app
USER cortex

ENV PYTHONPATH=/app
ENV CORTEX_ENV=production

# Default command
CMD ["python3", "-m", "cortex.main"]