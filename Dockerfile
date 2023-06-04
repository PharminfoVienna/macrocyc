FROM continuumio/miniconda3:latest
WORKDIR /app
RUN conda install -y -c conda-forge pandas plotly==5.9.0 dash==2.6.0 gunicorn && pip install dash-bootstrap-components==1.2.0
COPY ./ /app
EXPOSE 8050
ENTRYPOINT ["python", "main.py"]