FROM nvidia/cuda
FROM tensorflow/tensorflow:latest-gpu
ADD main.py main.py

RUN python3 main.py