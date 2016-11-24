#!/usr/bin/env bash

docker run -v ${PWD}/travis -it kaiszuttor/kaipy:latest /bin/bash -c "pip install codecov --user && cd travis && pip install . --user && coverage run setup.py test && codecov"
