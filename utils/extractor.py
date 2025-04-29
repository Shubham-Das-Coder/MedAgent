# utils/extractor.py
import torch

def extract_medical_info(model, tokenizer, text, device):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    # Safely get logits only if they exist
    logits = getattr(outputs, "logits", None)
    if logits is not None:
        predictions = torch.argmax(logits, dim=-1)
        return tokenizer.decode(predictions[0], skip_special_tokens=True)
    return "No logits found in model output."
