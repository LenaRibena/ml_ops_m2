import os

from fastapi.testclient import TestClient

from m2.api import app


def test_read_root():
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200, "Wrong status code for root path"
        assert response.json() == {"message": "OK", "status_code": 200}, "Wrong response for root path"

def test_read_item():
    with TestClient(app) as client:
        response = client.get("/items/1")
        assert response.status_code == 200, "Wrong status code for items path"
        assert response.json() == {"item_id": 1}, "Wrong response for items path"
    
    # Invalid item_id
    with TestClient(app) as client:
        response = client.get("/items/invalid")
        assert response.status_code == 422, "Wrong status code for items path"
        assert response.json() == {
            "detail":
                [
                    {"type":"int_parsing",
                     "loc":["path","item_id"],
                     "msg":"Input should be a valid integer, unable to parse string as an integer",
                     "input":"invalid"
                     }
                    ]
                }, "Wrong response for items path"

def test_read_restricted_item():
    with TestClient(app) as client:
        response = client.get("/restrict_items/alexnet")
        assert response.status_code == 200, "Wrong status code for restricted items path"
        assert response.json() == {"item_id": "alexnet"}, "Wrong response for restricted items path"
    
    # Invalid item_id
    with TestClient(app) as client:
        response = client.get("/restrict_items/invalid")
        assert response.status_code == 422, "Wrong status code for restricted items path"
        assert response.json() == {
            "detail":
                [
                    {"type":"enum",
                     "loc":["path","item_id"],
                     "msg":"Input should be 'alexnet', 'resnet' or 'lenet'",
                     "input":"invalid",
                     "ctx":{"expected":"'alexnet', 'resnet' or 'lenet'"}
                     }
                    ]
                }, "Wrong response for restricted items path"

def test_read_query_items():
    with TestClient(app) as client:
        response = client.get("/query_items?item_id=1")
        assert response.status_code == 200, "Wrong status code for query items path"
        assert response.json() == {"item_id": 1}, "Wrong response for query items path"
    
    # Invalid item_id
    with TestClient(app) as client:
        response = client.get("/query_items?item_id=invalid")
        assert response.status_code == 422, "Wrong status code for query items path"
        assert response.json() == {
            "detail":
                [
                    {"type":"int_parsing",
                     "loc":["query","item_id"],
                     "msg":"Input should be a valid integer, unable to parse string as an integer",
                     "input":"invalid"
                     }
                    ]
                }, "Wrong response for query items path"

def test_login():
    with TestClient(app) as client:
        response = client.post("/login/", params={"username": "user", "password": "password"})
        assert response.status_code == 200, "Wrong status code for login path"
        assert response.json() == {"message": "Login saved", "status_code": 200}, "Wrong response for login path"
    
    # Check that the login was saved
    with open('database.csv', "r") as file:
        lines = file.readlines()
        assert len(lines) == 1, "Login was not saved"
        assert lines[0] == "user, password \n", "Wrong login saved"
    
    # Check that the login was not saved
    with TestClient(app) as client:
        response = client.post("/login/", params={"username": "user", "password": "password"})
    
    with open('database.csv', "r") as file:
        lines = file.readlines()
        assert len(lines) == 1, "Login was saved again"
        assert lines[0] == "user, password \n", "Wrong login saved again"
    
    # Invalid request
    with TestClient(app) as client:
        response = client.post("/login/", params={"username": "user"})
        assert response.status_code == 422, "Wrong status code for login path"
        assert response.json() == {
            'detail': 
                [
                    {'type': 'missing', 
                     'loc': ['query', 'password'], 
                     'msg': 'Field required', 'input': None
                     }
                    ]
                }, "Wrong response for login path"
    
def test_contains_email():
    with TestClient(app) as client:
        response = client.get("/text_model/", params={"data": "mail@mail.com"})
        assert response.status_code == 200, "Wrong status code for text model path"
        assert response.json() == {
            "input": "mail@mail.com",
            "message": "OK",
            "status_code": 200,
            "is_email": True
        }, "Wrong response for contains email path"
    
    # Invalid email
    with TestClient(app) as client:
        response = client.get("/text_model/", params={"data": "mail@mail"})
        assert response.status_code == 200, "Wrong status code for text model path"
        assert response.json() == {
            "input": "mail@mail",
            "message": "OK",
            "status_code": 200,
            "is_email": False
        }, "Wrong response for contains email path"

def test_contains_email_domain():
    with TestClient(app) as client:
        response = client.post("/text_model/", json={"email": "mail@gmail.com", "domain": "gmail"})
        assert response.status_code == 200, "Wrong status code for text model path"
        assert response.json() == {
            "input": {"email": "mail@gmail.com", "domain": "gmail"},
            "message": "OK",
            "status_code": 200,
            "is_email": True
        }, "Wrong response for contains email domain path"
    
    # Invalid domain
    with TestClient(app) as client:
        response = client.post("/text_model/", json={"email": "mail@yahoo.com", "domain": "yahoo"})
        assert response.status_code == 200, "Wrong status code for text model path"
        assert response.json() == {
            "input": {"email": "mail@yahoo.com", "domain": "yahoo"},
            "message": "OK",
            "status_code": 200,
            "is_email": False
        }, "Wrong response for contains email domain path"
    
def test_cv_model():
    with TestClient(app) as client:
        # Valid image upload
        with open("resized_image.jpg", "rb") as img:
            response = client.post("/cv_model/", files={"data": img})
            assert response.status_code == 200, "Wrong status code for valid image upload"
            assert response.json()["message"] == "OK", "Wrong response message for valid image upload"

        # Invalid image upload
        with open("perleplade.py", "rb") as img:
            response = client.post("/cv_model/", files={"data": img})
            assert response.status_code == 500, "Wrong status code for invalid image upload"
            assert response.json()["detail"] == 'Image processing error: 400: Invalid image', "Wrong response detail for invalid image upload"
        
        # Missing file
        response = client.post("/cv_model/")
        assert response.status_code == 422, "Wrong status code for missing file"
        assert response.json() == {
            'detail': [
                {
                    'input': None, 
                    'loc': ['body', 'data'], 
                    'msg': 'Field required', 
                    'type': 'missing'
                    }
                ]
            }, "Wrong response for missing file"
        
if __name__ == "__main__":
    import pdb
    pdb.set_trace()