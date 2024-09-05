from abc import ABC, abstractmethod
from yaraa.models.base import Generator, Encoder
from yaraa.models.vectordb import Chroma, VectorDB, SUPPORTED_VECTOR_DATABASES

DEFAULT_PROMPT = '''
answer the following question based on the provided context
question: {question}
context: {context}
'''


class _RAG(ABC):
    @abstractmethod
    def generate(self, q: str, k: int = 3) -> tuple[str, list[tuple[str, float, dict]]]:
        raise NotImplementedError("you can't use this method in this class")


class Basic_RAG(_RAG):
    def __init__(self,
                 vectordb: str,
                 vectordb_path: str,
                 encoder_name_or_path: str,
                 generator_name_or_path: str, is_ollama: bool = False, is_server: bool = False, file_name: str = '',
                 prompt: str = '') -> None:

        self.encoder = Encoder(encoder_name_or_path)
        self.generator = Generator(
            generator_name_or_path, is_ollama, is_server, file_name)
        self.vectordb = SUPPORTED_VECTOR_DATABASES[vectordb](
            vectordb_name=vectordb, vectordb_path=vectordb_path, embedding_model=self.encoder)
        if prompt:
            self.prompt = prompt
        else:
            self.prompt = DEFAULT_PROMPT

    def generate(self, q: str, k: int = 3) -> tuple[str, list[tuple[str, float, dict]]]:
        results = self.vectordb.get_similiar_results_cossim(q=q, k=k)
        results_str = [chunk[0] for chunk in results]
        results_string = "".join(results_str)
        # do i need to affect in a new prompt?
        prompt = self.prompt.format(question=q, context=results_string)
        response = self.generator.generate(prompt)
        return response, results
