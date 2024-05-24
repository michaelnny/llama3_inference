# FastAPI API server for openAI API compatabilities


To start build the container for the FastAPI server, run the following command on the host machine.

```bash

cd api_server

docker-compose build


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