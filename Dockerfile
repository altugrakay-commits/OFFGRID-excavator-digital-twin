# ------------------------------------------------------------------ #
# Off-Grid Excavator Digital Twin — container image
#
# Build:
#   docker build -t offgrid-twin .
#
# Run (normal):
#   docker run --rm -p 8501:8501 offgrid-twin
#
# Run (off-grid proof — no network interfaces at all):
#   docker run --rm --network none -p 127.0.0.1:8501:8501 offgrid-twin
# ------------------------------------------------------------------ #

FROM python:3.11-slim

# --- System dependencies ---
# gcc        : to compile libstress.so from c_src/stress_solver.c
# libgl*     : OpenGL runtime required by PyVista / VTK
# libx*      : X11 client libraries required for off-screen rendering
# xvfb       : virtual framebuffer so PyVista can render headlessly
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        libgl1-mesa-glx \
        libglu1-mesa \
        libxrender1 \
        libxext6 \
        libx11-6 \
        xvfb \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- Python dependencies (cached layer) ---
COPY requirements.txt .
RUN pip install -no-cache-dir -r requirements.txt

# --- Source tree ---
COPY . .

# --- Compile the C solver inside the image ---
RUN python build.py

# -- Writable output directory ---
RUN mkdir -p output/cloud_spool

# --- Streamlit defaults ---
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 8501

# xvfb-run gives PyVista a display for off-screen rendering.
CMD [ "xvfb-run", "-a", "-s", "-screen 0 1280x800x24", \
    "streamlit", "run", "src/edge/dashboard.py"]