
FROM osgeo/gdal:ubuntu-small-latest

ENV DEBIAN_FRONTEND=noninteractive \
	PYTHONDONTWRITEBYTECODE=1 \
	PYTHONUNBUFFERED=1

WORKDIR /app

# Install Python 3.11 on top of the GDAL image.
RUN apt-get update \
	&& apt-get install -y --no-install-recommends \
		ca-certificates \
		curl \
		software-properties-common \
	&& add-apt-repository ppa:deadsnakes/ppa \
	&& apt-get update \
	&& apt-get install -y --no-install-recommends \
		python3.11 \
		python3.11-venv \
	&& curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 \
	&& python3.11 -m pip install --no-cache-dir --upgrade pip \
	&& rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN python3.11 -m pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

CMD ["python3.11", "-c", "import ndvi_calculator; print('ndvi_calculator ready')"]

