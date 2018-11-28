FROM tiangolo/uwsgi-nginx-flask:python3.6

ENV NGINX_WORKER_PROCESSES 12

COPY ./app /app

RUN pip3 --no-cache-dir install --upgrade Flask Pillow requests numpy