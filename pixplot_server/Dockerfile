FROM python:3.9-slim-buster

RUN apt-get update && \
    apt-get -y install build-essential

# Install Nginx
RUN apt-get -y update && apt-get -y install nginx
RUN ln -sf /dev/stdout /var/log/nginx/access.log && ln -sf /dev/stderr /var/log/nginx/error.log
COPY nginx/default /etc/nginx/sites-available/default


WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . .

RUN ["chmod", "+x", "/app/startup.sh"]

ENTRYPOINT ["./startup.sh"]
CMD ["/data/", "8600", "metadata.csv", "images/*.jpg", "False"]