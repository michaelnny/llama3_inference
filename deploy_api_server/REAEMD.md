# FastAPI API server for openAI API compatabilities

## Build Docker container

To start build the container for the FastAPI server, run the following command on the host machine.

```bash

cd deploy_api_server

docker-compose build

```

## Start API server

We can use the following command to start the container

```bash

cd deploy_api_server

docker-compose up

# Or run in detached mode
docker-compose up -d

# Incase want to open interactive command, here 'api-server' is the service name inside the 'docker-compose.yaml' file
docker-compose run --rm api-server

```

## Test the API server

We can then test the API server by run the following command in the host machine

```bash

curl http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
     "model": "llama3",
     "messages": [{"role": "user", "content": "Tell me a short joke about dogs"}],
     "temperature": 0.7,
     "stream": true
   }'


```

Test embedding endpoint

```bash

curl http://localhost:3000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
     "model": "st_ensemble",
     "input": "Tell me a short joke about dogs"
   }'


```

## Test OpenAI python Client

Our API server should also support requests from the native OpenAI python Client.

We can use the following script to test OpenAI python Client

```bash

cd deploy_api_server

python3 scripts/test_api_completion.py


python3 scripts/test_api_completion.py --stream



# For embedding

python3 scripts/test_api_embedding.py

```

## Test with Demo Chat

To further test the API server's compatibilities, we can use a very simple streamlit UI to check the connection and openAI API compatibilities.

On your host machine, install streamlit and then launch the demo UI:

```bash

pip3 install streamlit openai


cd deploy_api_server


streamlit run scripts/demo_chat.py

```
