# Docker version 27.3.1
FROM ultralytics/ultralytics:8.3.28

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt && pip freeze > reqs.txt

COPY model/ model/

COPY src/ src/

# EXPOSE 80/tcp

CMD [ "python", "src/main.py" ]

# CMD ["fastapi", "run", "app/main.py", "--port", "80", "--workers", "2"]
