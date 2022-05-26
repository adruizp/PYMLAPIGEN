# syntax=docker/dockerfile:1
FROM python:3.10.4-slim-buster
WORKDIR /pymlapigen
ADD . /pymlapigen
RUN export FLASK_APP=pymlapigen && \
    pip3 install -e .
CMD [ "python", "run.py" , "0.0.0.0"]