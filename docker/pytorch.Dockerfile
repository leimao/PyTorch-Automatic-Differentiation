FROM docker.io/pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel

ENV DEBIAN_FRONTEND noninteractive

RUN pip install functorch==0.1.0
