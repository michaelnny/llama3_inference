FROM python:3.10

# # Set environment variables for proxy (if needed), here I have proxy server running on my host machine on port 1081
# ENV https_proxy=http://172.17.0.1:1081
# ENV http_proxy=http://172.17.0.1:1081


COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# # Undo set environment variables for proxy
# ENV https_proxy=''
# ENV http_proxy=''


# Set the working directory
WORKDIR /app


# Copy project from host machine to container
COPY ./app /app


# Expose any necessary ports (if needed)
EXPOSE 3000


# Default command to run when container starts
CMD ["python3", "main.py", "--port", "3000", "--triton-server-url", "172.17.0.1:8001"]
