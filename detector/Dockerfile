ARG TARGETARCH
ARG ULTRALYTICS_TAG_AMD64=latest
ARG ULTRALYTICS_TAG_ARM64=latest-nvidia-arm64

FROM ultralytics/ultralytics:${ULTRALYTICS_TAG_AMD64} AS runtime-amd64
FROM ultralytics/ultralytics:${ULTRALYTICS_TAG_ARM64} AS runtime-arm64
FROM runtime-${TARGETARCH} AS runtime

# Install uv for fast package installation
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy application files
COPY . /app

# Write version file with REF_NAME (defaults to "main" if not provided)
ARG REF_NAME=main
RUN echo "TYPE = \"cuda\"" > /app/src/aidetector/utils/version.py
RUN echo "REF_NAME = \"${REF_NAME}\"" >> /app/src/aidetector/utils/version.py

# Lock torch versions to prevent replacement by system packages
RUN pip freeze | grep -E "^(torch|torchvision|torchaudio)" > constraints.txt

# Install the application directly into the ultralytics Python environment.
# amd64 images install the ONNX runtime stack explicitly; arm64 / Jetson images
# keep the runtime stack provided by the Ultralytics base image.
# --break-system-packages is needed because Debian marks system Python as externally managed
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ "$TARGETARCH" = "amd64" ]; then \
        uv pip install --system --break-system-packages --constraint constraints.txt ".[default,identity]"; \
    else \
        uv pip install --system --break-system-packages --constraint constraints.txt ".[identity]"; \
    fi

# Setup a non-root user
RUN groupadd --system --gid 999 nonroot \
    && useradd --system --gid 999 --uid 999 --create-home nonroot

# Give nonroot user write access to ultralytics weights directory (for model downloads)
RUN mkdir -p /ultralytics/weights && chown -R nonroot:nonroot /ultralytics

# Change ownership of the app directory and runtime data directory
RUN mkdir -p /data && chown -R nonroot:nonroot /app /data

# Use the non-root user to run our application
USER nonroot
WORKDIR /data

# Run the application
CMD ["python3", "/app/src/aidetector/__init__.py"]
