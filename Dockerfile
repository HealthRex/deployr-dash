FROM python:3.8.12

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

WORKDIR /src
COPY main.py /src
COPY utils /src/utils
COPY pages /src/pages
COPY components /src/components
COPY assets /src/assets

EXPOSE 8080

CMD ["streamlit", "run", "main.py"]