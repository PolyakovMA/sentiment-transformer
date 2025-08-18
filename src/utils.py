import os

def save_model(trainer, path):
    os.makedirs(path, exist_ok=True)
    trainer.save_model(path)