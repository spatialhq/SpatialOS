# Full UI image: base pipeline + Gradio web app + AI refinement stage.
# Large (pulls in torch + transformers) -- this is the image to use for the
# upload -> process -> refine -> visualize flow. For headless/CLI-only use,
# the default Dockerfile is much lighter.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libgomp1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies from a throwaway package stub first, so this layer
# only depends on pyproject.toml -- later `COPY src` changes (an actual code
# edit) don't force redownloading the whole torch/transformers stack.
COPY pyproject.toml README.md /app/
RUN mkdir -p src/spatial_os && touch src/spatial_os/__init__.py \
    && pip install --no-cache-dir -e ".[dev,app,ai]" \
    && rm -rf src

COPY src /app/src
COPY tests /app/tests
COPY scripts /app/scripts
COPY docs /app/docs

EXPOSE 7860
ENTRYPOINT ["spatial-os"]
CMD ["ui", "--port", "7860"]
