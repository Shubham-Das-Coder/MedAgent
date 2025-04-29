# utils/summarizer.py

from transformers import T5Tokenizer, T5ForConditionalGeneration

def summarize_text(model, tokenizer, text, device):
    model_name = model.__class__.__name__.lower()

    if "t5" in model_name:
        inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True).to(device)
        summary_ids = model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
        return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    elif "causallm" in model_name:
        prompt = f"Summarize the following:\n{text}\nSummary:"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
        outputs = model.generate(**inputs, max_length=200, num_beams=5, early_stopping=True)
        return tokenizer.decode(outputs[0], skip_special_tokens=True).replace(prompt, "").strip()

    else:
        # Fallback to classification models
        return "Model type not suitable for summarization."
