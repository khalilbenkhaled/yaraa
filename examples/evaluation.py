from yaraa.evaluation.metrics import Evaluator
from yaraa.RAG.RAG import Basic_RAG
from yaraa.models.base import Generator


# NOTE: evaluating with data
data = ['who is alexander', 'what is the university mentioned in the context']
r = Basic_RAG(
    vectordb='chroma',
    vectordb_path='./db',
    encoder_name_or_path='sentence-transformers/all-mpnet-base-v2',
    generator_name_or_path='qwen2:0.5b-instruct-q4_K_M', is_ollama=True,
)
judge = Generator(
    model_name_or_path='qwen2:0.5b-instruct-q4_K_M', is_ollama=True)

e = Evaluator(['answer relevance'], r, judge)
score = e.evaluate(data)
