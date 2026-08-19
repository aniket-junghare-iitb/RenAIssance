FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    git \
    curl \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libfontconfig1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Install Python dependencies (CPU-only)
COPY requirements.txt .
RUN pip3 install --no-cache-dir typing-extensions>=4.10.0 && \
    pip3 install --no-cache-dir torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu && \
    pip3 install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p weights basenet/quantized basenet/ocr_weights

# Copy Python module needed for CRAFT
COPY code/CRAFT/basenet/__init__.py basenet/
COPY code/CRAFT/basenet/vgg16_bn.py basenet/

# Copy only processor configuration files (skip the large model.safetensors)
COPY models/config.json basenet/ocr_weights/
COPY models/generation_config.json basenet/ocr_weights/
COPY models/preprocessor_config.json basenet/ocr_weights/
COPY models/special_tokens_map.json basenet/ocr_weights/
COPY models/tokenizer_config.json basenet/ocr_weights/
COPY models/tokenizer.json basenet/ocr_weights/
COPY models/vocab.json basenet/ocr_weights/
COPY models/merges.txt basenet/ocr_weights/

# Copy just the quantized ONNX model files
COPY quantized_model/ basenet/quantized/

# Copy application code and CRAFT weights
COPY code/app/qapp.py .
COPY code/CRAFT/craft.py .
COPY code/CRAFT/craft_utils.py .
COPY code/CRAFT/imgproc.py .
COPY code/CRAFT/refinenet.py .
COPY weights/ ./weights/

# Expose Streamlit port
EXPOSE 8501

# Create a healthcheck
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Create a startup script
RUN cat > /app/start.sh << 'EOL'
#!/bin/bash
echo "Starting CPU-Optimized OCR application with quantized ONNX model..."

# Display some system information
echo "System information:"
echo "- Python: $(python --version)"
echo "- CPU: $(grep "model name" /proc/cpuinfo | head -1 | cut -d':' -f2 | xargs)"
echo "- Memory: $(grep MemTotal /proc/meminfo | awk '{print $2/1024/1024 " GB"}')"

# Verify model paths
echo "Checking model paths..."
if [ -d "basenet/quantized" ]; then
    echo "✅ ONNX quantized model directory found"
    ls -la basenet/quantized
else
    echo "❌ Warning: basenet/quantized directory not found"
fi

if [ -d "basenet/ocr_weights" ]; then
    echo "✅ TrOCR processor directory found"
    ls -la basenet/ocr_weights
else
    echo "❌ Warning: basenet/ocr_weights directory not found"
fi

if [ -d "weights" ]; then
    echo "✅ CRAFT model weights directory found"
    ls -la weights
else
    echo "❌ Warning: weights directory not found"
fi

echo "Starting Streamlit server..."
streamlit run qapp.py --server.port=8501 --server.address=0.0.0.0
EOL
RUN chmod +x /app/start.sh

# Set entry point to our startup script
ENTRYPOINT ["/app/start.sh"]