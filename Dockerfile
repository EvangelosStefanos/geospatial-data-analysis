# Docker version 27.3.1
FROM python:3.9

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

RUN pip freeze > reqs.txt

COPY src/ src/

EXPOSE 80/tcp

CMD [ "python", "src/main.py" ]

# CMD ["fastapi", "run", "app/main.py", "--port", "80", "--workers", "2"]
