FROM python:3.11-slim
WORKDIR /code
COPY ./requirements_api.txt /code/requirements_api.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements_api.txt
COPY src/m2/api /code/app

CMD ["uvicorn", "app", "--host", "0.0.0.0", "--port", "80"]