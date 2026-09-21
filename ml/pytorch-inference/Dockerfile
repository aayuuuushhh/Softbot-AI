# CPU PyTorch inference for fine-tuned xView2 checkpoints
# Build after training: docker build -t darknem-xview2-pytorch -f Dockerfile ..
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    git libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY pytorch-xview2 /workspace/pytorch-xview2
COPY pytorch-inference /workspace/pytorch-inference

RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir \
      pillow numpy opencv-python-headless shapely tqdm joblib \
      pytorch-lightning==1.9.5 albumentations fire monai scipy scikit-image

ENV PYTHONPATH=/workspace/pytorch-xview2
ENV CHECKPOINT_PATH=/checkpoints/best.ckpt

ENTRYPOINT ["python", "/workspace/pytorch-inference/infer_pair.py"]
