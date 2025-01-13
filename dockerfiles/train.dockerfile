# Base image
FROM python:3.11-slim

# Install Python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

# We need each of these files to 
#   1) install necessary packages
#   2) install the package itself including all metadata and dependencies
#   3) copy the source code itself
#   4) copy the data files
COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY src/m2/ src/m2/
COPY data/ data/
COPY models/ models/
COPY reports/ reports/

# Set up working directory to root. We use --no-cache-dir to make the Docker image smaller
WORKDIR /
RUN --mount=type=cache,target=~/pip/.cache pip install -r requirements.txt --no-cache-dir
RUN pip install . --no-deps --no-cache-dir

# Run the training script
ENTRYPOINT ["python", "-u", "src/m2/train.py", "--epochs", "2"]
