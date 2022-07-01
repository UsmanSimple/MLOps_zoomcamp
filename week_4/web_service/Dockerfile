FROM python:3.9.7-slim

RUN pip install -U pip
RUN pip install pipenv 

WORKDIR /app

COPY [ "Pipfile", "Pipfile.lock", "./" ]

RUN pipenv install --system --deploy

COPY [ "predict.py", "lin_reg.bin", "./" ]

EXPOSE 8080

ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:8080", "predict:app" ]