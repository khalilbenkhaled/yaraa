from yaraa.RAG.RAG import Basic_RAG
prompt = '''
answer the following question.
question: {question}
context: {context}
'''
r = Basic_RAG(
    vectordb='chroma',
    vectordb_path='./db',
    encoder_name_or_path='sentence-transformers/all-mpnet-base-v2',
    generator_name_or_path='qwen2:0.5b-instruct-q4_K_M', is_ollama=True,
    prompt=prompt
)

q = 'who is alexandra thomson'
reply = r.generate(q=q)
