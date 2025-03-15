import torch
import openai
import vec2text

def get_embeddings_openai(text_list, model="text-embedding-ada-002") -> torch.Tensor:
    client = openai.OpenAI()
    response = client.embeddings.create(
        input=text_list,
        model=model,
        encoding_format="float",  # override default base64 encoding...
    )
    outputs = [e["embedding"] for e in response["data"]]
    return torch.tensor(outputs)

device = "cuda" if torch.cuda.is_available() else "cpu"

input_text = [
    'It was the age of incredulity, the age of wisdom, the age of apocalypse, the age of apocalypse, it was the age of faith, the age of best faith, it was the age of foolishness'
]

embeddings = get_embeddings_openai(input_text).to(device)

result = vec2text.invert_embeddings(
    embeddings=embeddings,
    corrector=corrector
)

print(result)
