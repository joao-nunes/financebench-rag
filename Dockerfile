# ------------------------------------------------------------
# Base image
# ------------------------------------------------------------
FROM python:3.11-slim

# ------------------------------------------------------------
# Prevent Python from writing .pyc files
# and force stdout/stderr to be unbuffered
# ------------------------------------------------------------
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ------------------------------------------------------------
# Set working directory
# ------------------------------------------------------------
WORKDIR /app

# ------------------------------------------------------------
# Install system dependencies (if needed)
# ------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy packaging files
COPY pyproject.toml README.md ./

# Copy source code
COPY src ./src
COPY api ./api


# ------------------------------------------------------------
# Install Python dependencies
# ------------------------------------------------------------
RUN pip install --no-cache-dir --upgrade pip

# ------------------------------------------------------------
# Copy the rest of the application
# ------------------------------------------------------------
COPY . .

RUN pip install --no-cache-dir .

# ------------------------------------------------------------
# Expose FastAPI port
# ------------------------------------------------------------
EXPOSE 8000

# ------------------------------------------------------------
# Start the API
# ------------------------------------------------------------
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]