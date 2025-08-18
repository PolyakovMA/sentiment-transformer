from transformers import AutoTokenizer, AutoModelForSequenceClassification

def load_model(model_name='distilbert-base-uncased', num_labels=2):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    return tokenizer, model