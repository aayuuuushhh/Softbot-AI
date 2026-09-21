# xView2 baseline inference — patched for modern Docker/PyPI (TF 1.15 base image)
# Clones upstream baseline inside the image; no local ml/xview2-baseline checkout required.
FROM tensorflow/tensorflow:1.15.5-py3

ENV DEBIAN_FRONTEND=noninteractive

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl git python3-opencv \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -g 999 appuser 2>/dev/null || true && \
    useradd -r -u 999 -g appuser appuser 2>/dev/null || true && \
    mkdir -p /home/appuser/.keras/models && \
    chown -R appuser:appuser /home/appuser

ADD https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5 \
    /home/appuser/.keras/models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5

RUN chown -R appuser:appuser /home/appuser/ && chmod -R u+rwX /home/appuser/

USER appuser
WORKDIR /home/appuser

RUN mkdir -p ~/code/xview-2/ ~/submission/ ~/output/

# Use apt python3-opencv (no pip opencv — avoids 60+ min source compile)
# Pin keras 2.2.5 for TensorFlow 1.15 compatibility
RUN pip install --user \
    matplotlib tqdm libtiff scipy 'Pillow<10' scikit-image \
    IPython 'keras==2.2.5' simplification scikit-learn chainer \
    tensorboard tensorboardX lxml xmljson Shapely && \
    pip install --user --no-deps imantics imgaug && \
    (pip install --user 'geopandas<0.11' || pip install --user geopandas)

RUN git clone --depth 1 https://github.com/DIUx-xView/xview2-baseline.git ~/code/xview-2

RUN curl -L https://github.com/DIUx-xView/xView2/releases/download/v1.0/localization.h5 \
    -o ~/code/xview-2/weights/localization.h5 && \
    curl -L https://github.com/DIUx-xView/xView2/releases/download/v1.0/classification.hdf5 \
    -o ~/code/xview-2/weights/classification.hdf5

RUN chmod +x ~/code/xview-2/submission/run.sh

ENTRYPOINT ["/home/appuser/code/xview-2/submission/run.sh", "/home/appuser/code/xview-2/"]
