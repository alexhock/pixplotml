https://www.baeldung.com/linux/nginx-docker-container

docker build . -t pp

docker run -d -p 80:80 pp/server
