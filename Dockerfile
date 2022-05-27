# syntax=docker/dockerfile:1
FROM python:3.10.4-slim-buster
WORKDIR /pymlapigen
ADD . /pymlapigen
RUN pip install -r requirements.txt
CMD [ "python", "run.py" , "0.0.0.0"]