#base
FROM python:3.11-slim-buster

#workdir
WORKDIR /app

#copy
COPY . /app

#run


#port
RUN pip install -r requirements.txt

#command
CMD ["python3","app.py"]