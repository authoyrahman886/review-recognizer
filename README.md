# Review Recognizer

A sentiment classifier that predicts whether a movie review is **positive** or **negative**, built by fine-tuning `distilbert-base-uncased` on the IMDb movie reviews dataset.

## What it does

Takes a piece of text (originally movie reviews) and classifies it as positive or negative sentiment, along with a confidence score.

## How it was built

- **Base model:** `distilbert-base-uncased` (a pretrained language model)
- **Dataset:** [IMDb movie reviews](https://huggingface.co/datasets/stanfordnlp/imdb) — 2,000 training examples, 500 evaluation examples
- **Method:** Fine-tuning using Hugging Face's `transformers` library and `Trainer` API
- **Training environment:** Google Colab (GPU)

## Results

| Metric   | Score |
|----------|-------|
| Accuracy | *0.888* |
| F1       | *0.889763779527559* |

## Try it

The fine-tuned model is hosted on Hugging Face and can be used directly:

```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis", model="authoy6788/review-recognizer")
result = classifier("This movie was fantastic!")
print(result)
```

Model page: [huggingface.co/authoy6788/review-recognizer](https://huggingface.co/authoy6788/review-recognizer)

## What I learned

This was my first end-to-end fine-tuning project — covering how to load a pretrained model, prepare and tokenize a dataset, fine-tune it on a specific task, evaluate performance (accuracy/F1), and test it on out-of-domain text to understand its limitations (e.g., confidence dropping sharply on text outside its training domain, like unrelated sentences or a different language).

## Files

- `review_recognizer.ipynb` — full training notebook, including code, training logs, and test predictions

## Limitations

Trained on a small subset of English-language movie reviews. Performance is unreliable on out-of-domain text (different topics, writing styles, or languages).
