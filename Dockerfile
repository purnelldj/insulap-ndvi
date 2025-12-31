FROM ghcr.io/osgeo/gdal:ubuntu-small-3.10.0

ENV DEBIAN_FRONTEND=noninteractive \
	PYTHONDONTWRITEBYTECODE=1 \
	PYTHONUNBUFFERED=1

WORKDIR /app

# Install pip for the base image's Python.
RUN apt-get update \
	&& apt-get install -y --no-install-recommends \
		python3-pip \
	&& rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN python3 -m pip install --no-cache-dir --break-system-packages -r /app/requirements.txt

COPY . /app

CMD ["python3", "-c", "import ndvi_calculator; print('ndvi_calculator ready')"]
