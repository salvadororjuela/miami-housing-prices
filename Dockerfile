FROM python:3.10

RUN python -m pip install --upgrade pip

WORKDIR /3-miami-housing-prices

COPY requirements.txt ./requirements.txt 

RUN pip install -r requirements.txt

EXPOSE 8501

COPY . /3-miami-housing-prices

ENTRYPOINT [ "streamlit", "run" ]

CMD ["miami_housing_prices.py"]