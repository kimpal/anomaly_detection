# just a small fastAPI file to let the docker haw a file to run to let the container be happy
from typing import Optional

from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"Welcome to the": "Anomaly-detection application"}