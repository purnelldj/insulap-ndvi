FROM ghcr.io/osgeo/gdal:ubuntu-small-3.10.0

ENV DEBIAN_FRONTEND=noninteractive \
	PYTHONDONTWRITEBYTECODE=1 \
	PYTHONUNBUFFERED=1

# Install pip for the base image's Python.
RUN apt-get update \
	&& apt-get install -y --no-install-recommends \
		python3-pip \
	&& rm -rf /var/lib/apt/lists/*

RUN mkdir -p /home/worker/processor

WORKDIR /home/worker/processor

COPY . /home/worker/processor

RUN chmod +x /home/worker/processor/workflow.sh

RUN python3 -m pip install --no-cache-dir --break-system-packages -r requirements.txt

ENTRYPOINT ["/home/worker/processor/workflow.sh"]