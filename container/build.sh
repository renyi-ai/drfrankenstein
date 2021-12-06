#!/usr/bin/env bash
source .dockerenv

# docker build --no-cache --rm -t ${IMAGE_NAME} .
docker build --rm -t ${IMAGE_NAME} .
