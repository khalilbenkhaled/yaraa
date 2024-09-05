from yaraa.models import base

# NOTE: testing the generation with the ollama
model = base.Generator(
    model_name_or_path='qwen2:0.5b-instruct-q4_K_M', is_ollama=True)
r = model.generate('1+1=?')
print(r)

# NOTE: testing the generation with the server
api = 'API_KEY'
url = 'https://glhf.chat/api/openai/v1'
model = base.Generator(
    model_name_or_path='hf:mistralai/Mistral-7B-Instruct-v0.3', is_server=True, url=url, api=api)
r = model.generate('1+1=?')


# NOTE: testing if the encoding works
encoder = base.Encoder('sentence-transformers/all-mpnet-base-v2')
r = encoder.encode('hello')
