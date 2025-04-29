# models/lite/model_loader.py

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModel
)

class ModelLoader:
    def __init__(self, model_names):
        self.model_names = model_names
        self.cache = {}

    def load_model(self, model_key, device):
        if model_key in self.cache:
            return self.cache[model_key]

        model_name = self.model_names[model_key]

        # Load BioGPT as CausalLM
        if model_key == "BioGPT":
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        elif "t5" in model_name.lower():
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name).to(device)

        self.cache[model_key] = (model, tokenizer)
        return model, tokenizer
