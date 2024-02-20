FROM python:3.10

WORKDIR /3-miami-housing-prices

COPY requirements.txt ./requirements.txt 

RUN pip install -r requirements.txt

EXPOSE 8501

COPY . /3-miami-housing-prices

ENV HOSTNAME=0.0.0.0

CMD streamlit run --server.port 8501 --server.enableCORS false Hello.py