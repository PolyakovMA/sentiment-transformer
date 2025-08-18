from src.data_preprocessing import load_dataset, preprocess_dataset
from src.model import load_model
from src.train import train_model
from src.evaluate import compute_metrics
from datasets import Dataset

# 1. Загрузка и предобработка
url = "https://raw.githubusercontent.com/aiedu-courses/all_datasets/refs/heads/main/Womens%20Clothing%20E-Commerce%20Reviews.csv"
dataset = load_dataset(url)
dataset = preprocess_dataset(dataset)

# 2. Токенизация и преобразование в Dataset HuggingFace
tokenizer, model = load_model()
tokenized = dataset['Review Text'].apply(lambda x: tokenizer(x, padding='max_length', truncation=True))
dataset['input_ids'] = tokenized.apply(lambda x: x['input_ids'])
dataset['attention_mask'] = tokenized.apply(lambda x: x['attention_mask'])
dataset = dataset.drop(columns='Review Text')

ds = Dataset.from_pandas(dataset).train_test_split(test_size=0.2)

# 3. Обучение
trainer = train_model(model, ds['train'], ds['test'], output_dir='./results', compute_metrics=compute_metrics)
