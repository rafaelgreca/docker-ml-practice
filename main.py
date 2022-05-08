import argparse
import pandas as pd
import sys

from model import LSTM_Model

# create the parser
parser = argparse.ArgumentParser()

# adding the arguments
parser.add_argument('-data_path', '--data_path', type=str, help='Project file root path', required=True)
parser.add_argument('-embedding_size', '--embedding_size', type=int, help='Embedding size', required=True)
parser.add_argument('-max_len', '--max_len', type=int, help='Max sentence size', required=True)
parser.add_argument('-dropout_rate', '--dropout_rate', type=float, help='Dropout rate', required=True)
parser.add_argument('-lstm_units', '--lstm_units', type=int, help='LSTM layer units', required=True)
parser.add_argument('-epochs', '--epochs', type=int, help='Total epochs training', required=True)
parser.add_argument('-batch_size', '--batch_size', type=int, help='Batch size training', required=True)

args = parser.parse_args()

if not len(sys.argv) > 1:
    raise Exception('No arguments passed.')
else:
    train = pd.read_csv('{}/data/train.csv'.format(args.data_path), sep=',')
    train['tweet'] = train['tweet'].apply(str)
    train['tweet'] = train['tweet'].apply(lambda x: x.lower())
    train = train.drop_duplicates()
    train = train.drop(columns=['id'])

    lstm = LSTM_Model(df=train, 
                      path=args.data_path,
                      embedding_size=args.embedding_size,
                      epochs=args.epochs,
                      max_len=args.max_len,
                      dropout_rate=args.dropout_rate,
                      batch_size=args.batch_size,
                      lstm_units=args.lstm_units)
    lstm.build()
    lstm.train()
    print('F1-Score validation: {}'.format(lstm.evaluate()))

    test = pd.read_csv('{}/data/test.csv'.format(args.data_path), sep=',')
    lstm.prediction(test_df=test)