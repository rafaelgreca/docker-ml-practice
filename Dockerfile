FROM continuumio/anaconda3

COPY . /usr/docker-practice

EXPOSE 5000

WORKDIR /usr/docker-practice

RUN conda env create -f environment.yaml

SHELL ["conda", "run", "--no-capture-output", "-n", "myenv", "/bin/bash", "-c"]

CMD python main.py --data_path '/usr/docker-practice' --epochs 100 --batch_size 200 --embedding_size 300 --lstm_units 64 --dropout_rate 0.25 --max_len 30