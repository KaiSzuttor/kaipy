#!/usr/bin/env bash

docker run -u kaipy -v ${PWD}:/travis -it kaiszuttor/kaipy:latest /bin/bash -c "pip install codecov --user && git clone /travis && cd travis && pip install . --user && /home/kaipy/.local/bin/coverage run setup.py test && codecov"
