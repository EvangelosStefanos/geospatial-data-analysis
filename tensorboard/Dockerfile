# Docker version 27.3.1
FROM tensorflow/tensorflow:2.18.0

WORKDIR /app

EXPOSE 6006/tcp

VOLUME [ "/app/outputs" ]

CMD [ "tensorboard", "--logdir", "outputs", "--bind_all" ]
