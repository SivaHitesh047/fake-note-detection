
FROM python:3.8.8
COPY . /myapp
WORKDIR /myapp
RUN pip install -r requirements.txt
EXPOSE 5000
CMD ["python3","app.py" ]