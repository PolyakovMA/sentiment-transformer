import pandas as pd
import re


# load dataset, select features
def load_dataset(url):
    df = pd.read_csv(url)
    dataset = df[['Review Text', 'Recommended IND']].copy()
    dataset.rename(columns={'Recommended IND': 'labels'}, inplace=True)
    return dataset


def clear_text(text):
  text = str(text).lower()
  
  # delete emoji
  text = text.encode('ascii', 'ignore').decode('ascii')

  # delete everytnig but letters and spacebar
  text = re.sub(r'[^A-Za-z\s]', '', text)

  # delete extra spacebar
  text = re.sub(r"\s+", " ", text).strip()

  return text


# apply clear_text func to all reviews
def preprocess_dataset(dataset):
    dataset['Review Text'] = dataset['Review Text'].apply(clear_text)
    return dataset