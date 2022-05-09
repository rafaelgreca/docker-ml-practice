# Containerized Machine Learning model with Docker

The idea behind this repository is to learn how to use Docker and Machine Learning together, therefore the model developed is very simple and doesn't has the best results.

## How to run

First of all you **MUST** have two files (train.csv and test.csv) within a "data" folder. The files used were collected from [Identify the Sentiment](https://datahack.analyticsvidhya.com/contest/linguipedia-codefest-natural-language-processing-1/) - a machine learning competition hosted by Analytics Vidhya. 

Build the docker image:
```bash
docker build -t docker_practice .
```

Run the image:
```bash
docker run --rm docker_practice
```