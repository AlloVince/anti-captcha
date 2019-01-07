FROM tensorflow/tensorflow:latest-py3

RUN mkdir -p ~/.pip && echo "[global]\nindex-url = http://mirrors.aliyun.com/pypi/simple/\n[install]\ntrusted-host = mirrors.aliyun.com" > ~/.pip/pip.conf
RUN apt-get update && apt-get install -y libsm6 libxrender1 libfontconfig1 libxext-dev

RUN mkdir -p /opt/htdocs/anti-captcha && cd /opt/htdocs/anti-captcha
COPY requirements.txt /opt/htdocs/anti-captcha
WORKDIR /opt/htdocs/anti-captcha
RUN pip install -r requirements.txt

RUN mkdir -p /opt/htdocs/anti-captcha/models
COPY api.py /opt/htdocs/anti-captcha
COPY train.py /opt/htdocs/anti-captcha
COPY gen_captcha.py /opt/htdocs/anti-captcha
COPY index.html /opt/htdocs/anti-captcha

EXPOSE 8000
CMD gunicorn api:api