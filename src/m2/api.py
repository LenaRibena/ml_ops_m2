import re
from contextlib import asynccontextmanager
from enum import Enum
from http import HTTPStatus

import cv2
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel


class ItemEnum(Enum):
    alexnet = "alexnet"
    resnet = "resnet"
    lenet = "lenet"

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Hello!")
    yield
    print("Goodbye!")

app = FastAPI(lifespan=lifespan)


@app.get("/")
def read_root():
    '''Health check'''
    return {'message': HTTPStatus.OK.phrase, 'status_code': HTTPStatus.OK}

@app.get("/items/{item_id}")
def read_item(item_id: int):
    return {"item_id": item_id}

@app.get("/restrict_items/{item_id}")
def read_restricted_item(item_id: ItemEnum):
    return {"item_id": item_id.value}

@app.get("/query_items")
def read_query_item(item_id: int):
    return {"item_id": item_id}

database = {'username': [], 'password': []}

@app.post("/login/")
def login(username: str, password: str):
    username_db = database['username']
    password_db = database['password']
    if username not in username_db and password not in password_db:
        with open('database.csv', "a") as file:
            file.write(f"{username}, {password} \n")
        username_db.append(username)
        password_db.append(password)
    return {"message": "Login saved", "status_code": HTTPStatus.OK}

@app.get("/text_model/")
def contains_email(data: str):
    regex = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    return {
        "input": data,
        "message": HTTPStatus.OK.phrase,
        "status_code": HTTPStatus.OK,
        "is_email": re.fullmatch(regex, data) is not None
    }

class Mail(BaseModel):
    email: str
    domain: str

@app.post("/text_model/")
async def contains_email_domain(data: Mail):
    regex = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    return {
        "input": data,
        "message": HTTPStatus.OK.phrase,
        "status_code": HTTPStatus.OK,
        "is_email": re.fullmatch(regex, data.email) is not None and data.domain in ['gmail', 'hotmail']
    }

@app.post("/cv_model/")
async def cv_model(data: UploadFile = File(...), h: int = 28, w: int = 28):
    try:
        # Save the image with a unique filename
        file_name = "image.jpg"
        with open(file_name, 'wb') as image:
            content = await data.read()
            image.write(content)

        # Process the image
        img = cv2.imread(file_name)
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image")

        res = cv2.resize(img, (h, w))

        # Save the resized image
        resized_file_name = f"resized_{file_name}"
        cv2.imwrite(resized_file_name, res)

        return {
            "input": data.filename,
            "resized_image": FileResponse(f'resized_{file_name}'),
            "message": HTTPStatus.OK.phrase,
            "status_code": HTTPStatus.OK,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image processing error: {str(e)}")