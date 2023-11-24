import json
import os

from fastapi import FastAPI, UploadFile, routing
from fastapi.middleware.cors import CORSMiddleware
from fastapi.routing import APIRoute
from fastapi.responses import FileResponse
from prometheus_fastapi_instrumentator import Instrumentator

from Connector.unpacker import extractor

app = FastAPI(debug=True)

Instrumentator().instrument(app).expose(app)

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/video")
async def video():
    if not os.path.exists("./videos"):
        os.mkdir("./videos")
    list_file = os.listdir("./videos")

    if len(list_file) == 0:
        return {"error": "video not found"}
    else:
        dictionary = dict()
        counter = 0
        dictionary = {f"VideoURL{counter}": f"/get_url_video/{i}" for counter, i in enumerate(list_file, start=1)}
        return dictionary


@app.post("/archive_extract")
async def unpacker(file: UploadFile):
    if not os.path.exists("./archive"):
        os.makedirs("./archive")
    file_path = "./archive" + "/" + file.filename

    try:
        with open(file_path, "wb") as f:

            f.write(file.file.read())
        extractor(file_path)

    except Exception as e:
        return {"directory_is_empty": e.args}
    return {file.filename: "success"}


static_router = routing.APIRouter(route_class=APIRoute)


@static_router.get("/get_url_video/{filename}")
def get_static_image(filename: str):
    file_path = "./videos" + "/" + filename
    return FileResponse(file_path)


app.include_router(static_router)
