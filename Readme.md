docker run -p 8888:80 -v /Users/allovince/opt/htdocs/anti-captcha:/var/www/html --name captcha --rm php:7-apache-gd
docker build -t allovince/php:7-apache-gd .
docker push allovince/php:7-apache-gd 

apt-get update && apt-get upgrade -y

apt-get install -y git htop apt-transport-https ca-certificates curl software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | apt-key add -
apt-key fingerprint 0EBFCD88
add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"
apt-get update && apt-get install -y docker-ce

apt-get install -y make build-essential libssl-dev zlib1g-dev libbz2-dev \
libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev \
xz-utils tk-dev libffi-dev liblzma-dev python-openssl
curl -L https://github.com/pyenv/pyenv-installer/raw/master/bin/pyenv-installer | bash
pyenv install 3.6.8
pyenv global 3.6.8

   
git clone https://github.com/AlloVince/anti-captcha.git
cd anti-captcha
git checkout dev
pip install -r requirements.txt
docker run -v $(pwd)/samples:/var/www/html/samples --restart always --name captcha1 -d allovince/php:7-apache-gd php gen_captcha.php
MAX_WORKERS=12 python train_stream.py

