# FastAPI API server for openAI API compatabilities

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