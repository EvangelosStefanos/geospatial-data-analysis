# Docker version 27.3.1
FROM ultralytics/ultralytics:8.3.28

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt && pip freeze > reqs.txt

RUN pip install gdown

RUN pip freeze > reqs.txt

COPY weights/ weights/

COPY src/ src/

# EXPOSE 80/tcp

CMD [ "python", "src/change_detection.py" ]
