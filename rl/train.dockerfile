FROM python:3.11-slim AS builder

# Install build tools for C and C++ extensions and X11 libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ninja-build \
    libsdl2-dev \
    xvfb \
    x11-utils \
    libx11-6 \
    libxext-dev \
    libxrender-dev \
    libxtst-dev \
    libfreetype-dev \
    libfontconfig1 \
    && rm -rf /var/lib/apt/lists/*

# Configures settings for the image.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_ROOT_USER_ACTION=ignore
WORKDIR /workspace

# Installs your dependencies.
RUN pip install -U pip
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM builder

# Add pygame display driver setup
ENV SDL_VIDEODRIVER=x11

# Copies your source files.
COPY src .
RUN rm -rf checkpoints/
RUN rm -rf runs/train_rl/*
