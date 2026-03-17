FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies needed for torch, sentence-transformers, chromadb
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files first (for layer caching)
COPY pyproject.toml .
COPY uv.lock .

# Install uv and sync dependencies
RUN pip install uv && uv sync --frozen

# Copy the rest of the application
COPY . .

# Create directories that will be needed at runtime
RUN mkdir -p vector_store eval_results data/docs

# Expose port 7860 (required by Hugging Face Spaces)
EXPOSE 7860

# Set environment variables
ENV PORT=7860
ENV PYTHONUNBUFFERED=1

# Start the FastAPI server
CMD ["uv", "run", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]
