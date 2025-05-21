from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
def tokenize(example):
    full_text = example["prompt"] + tokenizer.eos_token + example["completion"]
    return tokenizer(full_text, truncation=True, max_length=512)
