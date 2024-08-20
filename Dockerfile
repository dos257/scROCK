#FROM python:3.10.14-alpine
# On Alpine "Getting requirements to build wheel" for hdf5plugin==4.4.0 fails with:
# ImportError: cannot import name 'get_platform' from 'wheel.bdist_wheel' (/tmp/pip-build-env-5sfrigve/overlay/lib/python3.10/site-packages/wheel/bdist_wheel.py)

FROM python:3.10.14-slim-bookworm

WORKDIR /app
COPY requirements.txt /app/requirements.txt

RUN python3 -m pip install --requirement requirements.txt --no-deps

# Add scrock library files lately, to not update layer for requirements.txt
ADD . /app
RUN python3 -m pip install .

ENTRYPOINT ["python3", "-m", "scrock"]
