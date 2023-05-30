import torch

from transformers import AutoTokenizer, AutoModel
import re

from tqdm.auto import tqdm


def text_preprocess(text):
    text = re.sub("[“”«»″]", '"', text)
    text = re.sub("‘’", '"', text)
    text = re.sub("’", "'", text)
    text = re.sub("[—–]+", '-', text)
    text = re.sub("【", '(', text)
    text = re.sub("】", ')', text)
    text = re.sub("[\\\]+|/", ' ', text)
    text = re.sub("[\n\t\xa0]", ' ', text)
    text = re.sub("[\u200b\xad•]", '', text)
    text = re.sub('"+', '"', text)
    text = re.sub(" +", ' ', text)
    text = text.strip()
    return text


def get_name_labse_embs(model_name: str, sentences: list[str], device: str, batch_size: int = 32) -> list[str]:
    embds = []

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)

    for i in tqdm(range(0, len(sentences), batch_size)):
        batch = sentences[i:i + batch_size]

        encoded_input = tokenizer(batch, padding=True, truncation=True, max_length=256, return_tensors='pt').to(device)
        with torch.no_grad():
            model_output = model(**encoded_input)
        embeddings = model_output.pooler_output

        embeddings = torch.nn.functional.normalize(embeddings).detach().cpu()

        for _ in embeddings:
            embds.append(_.numpy())

    return embds
