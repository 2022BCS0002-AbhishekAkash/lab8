FROM python:3.10

WORKDIR /app

COPY . .

RUN pip install pandas scikit-learn flask

EXPOSE 5000

CMD ["python", "app/app.py"]