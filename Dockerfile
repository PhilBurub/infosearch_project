FROM python:3.11.6-bookworm
WORKDIR usr/src/app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY app .

CMD ["python", "webapp.py"]