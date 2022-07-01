FROM python:3.9

RUN pip install -U pip
RUN pip install pipenv
RUN pip --no-cache-dir install --upgrade awscli

WORKDIR /app

COPY [ "Pipfile", "Pipfile.lock", "./" ]

RUN pipenv install --system --deploy

COPY [ "starter.py", "regressor.bin", "./" ]

ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY
ARG AWS_DEFAULT_REGION

RUN aws configure set aws_access_key_id $AWS_ACCESS_KEY_ID \
&& aws configure set aws_secret_access_key $AWS_SECRET_ACCESS_KEY \
&& aws configure set default.region $AWS_DEFAULT_REGION

