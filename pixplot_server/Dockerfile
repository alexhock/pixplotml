FROM python:3.9-slim-buster

RUN apt-get update && \
    apt-get -y install build-essential

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . .

RUN ["chmod", "+x", "/app/startup.sh"]

ENTRYPOINT ["./startup.sh"]
CMD ["/data/", "8600", "metadata.csv", "images/*.jpg"]