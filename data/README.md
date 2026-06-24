# Data

`intents.csv` and `kb.json` in this folder are **small dev-fallback samples**
only — enough to smoke-test the pipeline locally without a trained model.

The real data is the [Bitext Customer Support dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset)
(~27 intents, ~26.8k examples), pulled directly from the Hugging Face Hub
inside `notebooks/01_intent_classification.ipynb`. That notebook also
auto-derives a real `kb.json` from the dataset's own responses — download it
from the notebook's Kaggle output and replace this file with it before
training/deploying for real.
