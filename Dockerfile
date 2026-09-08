FROM python:3.11-slim

# Open3D needs a handful of runtime libs even in headless/CPU mode.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libgomp1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies from a throwaway package stub first, so this layer
# only depends on pyproject.toml -- later `COPY src` changes (an actual code
# edit) don't force redownloading the whole dependency set.
COPY pyproject.toml README.md /app/
RUN mkdir -p src/spatial_os && touch src/spatial_os/__init__.py \
    && pip install --no-cache-dir -e ".[dev]" \
    && rm -rf src

COPY src /app/src
COPY tests /app/tests
COPY scripts /app/scripts
COPY docs /app/docs

ENTRYPOINT ["spatial-os"]
CMD ["--help"]
