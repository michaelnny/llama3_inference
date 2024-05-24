# FastAPI API server for openAI API compatabilities


To start build the container for the FastAPI server, run the following command on the host machine.

```bash

cd deploy_api_server

docker-compose build

```

Then start the server:

```bash

docker-compose up


# Or run in detached mode
docker-compose up -d


# Incase want to open interactive command, here 'api-server' is the service name inside the 'docker-compose.yaml' file
docker-compose run --rm api-server

```

```bash

curl http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
     "model": "ensemble",
     "messages": [{"role": "user", "content": "Tell me a short joke about dogs"}],
     "temperature": 0.7,
     "stream": true
   }'


```