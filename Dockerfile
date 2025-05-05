# Use the official Python image for the base image.
FROM --platform=linux/amd64 python:3.11-slim

# Set environment variables to make Python print directly to the terminal and avoid .pyc files.
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies required for the project.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    git \
    wget \
    unzip \
    libvips-dev \
    gnupg2 \
    && rm -rf /var/lib/apt/lists/*


# Install pipx.
RUN python3 -m pip install --no-cache-dir pipx \
    && pipx ensurepath

# Add poetry to the path
ENV PATH="${PATH}:/root/.local/bin"

# Install the latest version of Poetry using pipx.
RUN pipx install poetry

# Set the working directory. IMPORTANT: can't be changed as needs to be in sync to the dir where the project is cloned
# to in the codespace
WORKDIR /workspaces/tianshou

# Copy the pyproject.toml and poetry.lock files (if available) into the image.
COPY pyproject.toml poetry.lock* README.md /workspaces/tianshou/

RUN poetry config virtualenvs.create false
RUN poetry install --no-root --with dev

# The entrypoint will perform an editable install, it is expected that the code is mounted in the container then
# If you don't want to mount the code, you should override the entrypoint
ENTRYPOINT ["/bin/bash", "-c", "poetry install --with dev && poetry run jupyter trust notebooks/*.ipynb docs/02_notebooks/*.ipynb && $0 $@"]