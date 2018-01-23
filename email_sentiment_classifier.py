from encoder import Model
model = Model()
import pandas
from tqdm import tqdm

sentiment_neuron = 2388
import sys

if len(sys.argv) < 2:
    start = 0
else:
    start = int(sys.argv[1])

def calculate_sentiment(body, messageid=None, debug=False):
    # sentiment neuron
    sentiment = model.transform([body])[0, sentiment_neuron]
    if debug and messageid:
        print("id: {} -> calculated senitment: {}".format(messageid, sentiment))
    return sentiment

if __name__ == '__main__':
    df = pandas.read_csv('bodies_clean.csv')
    df["sentiment"] = 0.0
    for i in tqdm(range(start, df.shape[0]), initial=start):
        messageid = df.iloc[i, 1]
        body = df.iloc[i, 2]
        if pandas.isnull(body):
            df.loc[i,3] = 0.0
        else:
            df.iloc[i, 3] = calculate_sentiment(body, messageid, True)
        if i % 10000 == 0 and i != 0: # save every ten thousand classifications.
            print('checkpointing at {}'.format(i))
            df.to_csv('bodies_clean_sentiment-checkpoint-{}.csv'.format(i))
    df.to_csv('bodies_clean_sentiment.csv')
