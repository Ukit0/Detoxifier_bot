from transformers import T5ForConditionalGeneration, AutoTokenizer

base_model_name = 'sberbank-ai/ruT5-base'
model_name = 'SkolkovoInstitute/ruT5-base-detox'

tokenizer = AutoTokenizer.from_pretrained(base_model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

def paraphrase(text, model, tokenizer, n=None, max_length="auto", beams=3):
    texts = [text] if isinstance(text, str) else text
    inputs = tokenizer(texts, return_tensors="pt", padding=True)["input_ids"].to(
        model.device
    )
    if max_length == "auto":
        max_length = inputs.shape[1] + 10

    result = model.generate(
        inputs,
        num_return_sequences=n or 1,
        do_sample=True,
        temperature=1.0,
        repetition_penalty=10.0,
        max_length=max_length,
        min_length=int(0.5 * max_length),
        num_beams=beams
    )
    texts = [tokenizer.decode(r, skip_special_tokens=True) for r in result]

    if not n and isinstance(text, str):
        return texts[0]
    return texts


while True:
    print('Начало!\n')
    text = input()
    print(paraphrase(text, model, tokenizer))
