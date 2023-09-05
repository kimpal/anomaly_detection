# just a litel fastAPI file to let the docker haw a file to runn to let the container be happy
from typing import Optional

from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"Welcome to the": "Anomaly-detection aplication"}