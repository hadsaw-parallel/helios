FROM rocm/pytorch:rocm6.2_ubuntu22.04_py3.10_pytorch_release_2.3.0

WORKDIR /app

# System deps for astropy / folium / streamlit
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl libssl-dev ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Skip torch reinstall (already in ROCm base image), install everything else
RUN pip install --no-cache-dir $(grep -v '^torch' requirements.txt | grep -v '^#' | grep -v '^$' | grep -v '^vllm')

# Install vLLM ROCm build
RUN pip install --no-cache-dir vllm

COPY . .

# Clone Surya — weights mounted at runtime via volume
RUN git clone https://github.com/NASA-IMPACT/Surya /app/Surya

EXPOSE 8501 8000

CMD ["streamlit", "run", "dashboard/app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
