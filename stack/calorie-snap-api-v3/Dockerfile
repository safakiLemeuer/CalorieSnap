# BASE_IMAGE lets you build on your existing calorie-snap-api image, which
# already has fastapi and uvicorn. That avoids pip entirely on networks
# where SSL inspection blocks PyPI. Find the name with:
#   docker inspect calorie-snap-api --format '{{.Config.Image}}'
ARG BASE_IMAGE=python:3.12-slim
FROM ${BASE_IMAGE}
WORKDIR /srv/caloriesnap
COPY requirements.txt .
RUN python3 -c "import fastapi, uvicorn" 2>/dev/null \
    || pip install --no-cache-dir -r requirements.txt
COPY *.py ./
COPY static ./static
ENV PYTHONUNBUFFERED=1
EXPOSE 8021
ENTRYPOINT []
CMD ["python3", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8021"]
