
FROM python:3.10.14-slim-bookworm
 
WORKDIR /flask-docker
 
RUN python -m pip install --upgrade pip
COPY requirements.txt requirements.txt
Run pip install -r requirements.txt
 
COPY . .
 
CMD ["python3", "-m", "flask", "run", "--host=0.0.0.0"]