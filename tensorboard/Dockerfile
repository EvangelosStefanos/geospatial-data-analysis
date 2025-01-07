# Docker version 27.3.1
# Volume (bind-mount, read-only) at /app/output
FROM tensorflow/tensorflow:2.18.0

EXPOSE 6006/tcp

CMD [ "tensorboard", "--logdir", "/app/output", "--bind_all" ]
