FROM python:3.8.13-buster

ENV data_path='/usr/docker-practice'
ENV epochs=100
ENV batch_size=200
ENV embedding_size=300
ENV lstm_units=64
ENV dropout_rate=0.25
ENV max_len=30

COPY . /usr/docker-practice

EXPOSE 5000

WORKDIR /usr/docker-practice

RUN pip install -r requirements.txt

CMD python main.py --data_path $data_path --epochs $epochs --batch_size $batch_size --embedding_size $embedding_size --lstm_units $lstm_units --dropout_rate $dropout_rate --max_len $max_len