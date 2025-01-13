# Base image
FROM python:3.11-slim

# Install Python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Copy files
COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY src/m2/ src/m2/
COPY data/ data/

# Set up working directory
WORKDIR /
RUN --mount=type=cache,target=~/pip/.cache pip install -r requirements.txt --no-cache-dir
RUN pip install . --no-deps --no-cache-dir

# Run the evaluation script
ENTRYPOINT ["python", "-u", "src/m2/evaluate.py"]