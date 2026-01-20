FROM python:3.9-slim

# Install system dependencies for RDKit
RUN apt-get update && apt-get install -y \
    libxrender1 \
    libxext6 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create checkpoints directory if it doesn't exist (though it should be copied if populated)
RUN mkdir -p checkpoints

# Expose port (Huggging Face Spaces default)
EXPOSE 7860

# Run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
