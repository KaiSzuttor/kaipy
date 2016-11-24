#!/usr/bin/env bash

ci_env=`bash <(curl -s https://codecov.io/env)`
docker run $ci_env -u kaipy -v ${PWD}:/travis -it kaiszuttor/kaipy:latest /bin/bash -c "pip install codecov --user && git clone /travis && cd travis && pip install . --user && /home/kaipy/.local/bin/coverage run setup.py test && bash <(curl -s https://codecov.io/bash)"
