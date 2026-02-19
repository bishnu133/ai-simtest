FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml README.md ./
COPY src/ src/
COPY configs/ configs/

# Install package (non-editable for production)
RUN pip install --no-cache-dir ".[dev]"

# Download NLTK data for Presidio
RUN python -c "import nltk; nltk.download('punkt')" 2>/dev/null || true

EXPOSE 8000

# Default: run the API server
CMD ["simtest", "serve"]