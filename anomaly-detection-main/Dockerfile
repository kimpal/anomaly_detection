FROM python:3.10-slim
WORKDIR /usr/src
COPY requirements.txt ./
RUN apt-get update && apt-get install -y git
RUN pip install --upgrade pip \
    && pip install --no-cache-dir --upgrade -r requirements.txt

COPY .. ./anomaly-detection-main
EXPOSE 8888
CMD ["uvicorn", "anomaly-detection-main.app.main:app", "--host", "0.0.0.0", "--port", "8080", "--reload"]
WORKDIR /usr/src/anomaly-detection-main