from typing import Type

import random

from yaraa.models import base
from abc import ABC, abstractmethod

import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings


class VectorDB(ABC):

    def __init__(self, vectordb_name: str, vectordb_path: str, embedding_model: base.Encoder) -> None:
        if vectordb_name not in SUPPORTED_VECTOR_DATABASES:
            raise ValueError(
                f"the vector store you provided is not supported, here's the supported vector databases, you have to choose one of them: { ''.join(SUPPORTED_VECTOR_DATABASES)}")
        self.vectordb_path = vectordb_path
        self.embedding_model = embedding_model

    @abstractmethod
    def get_all(self) -> list[tuple[str, dict]]:
        raise NotImplementedError

    @abstractmethod
    def get_random(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_similiar_results_cossim(self, q: str, k: int = 3) -> list[tuple[str, float, dict]]:
        raise NotImplementedError


class Chroma(VectorDB):

    def __init__(self, vectordb_path: str, embedding_model: base.Encoder, vectordb_name='chroma') -> None:
        super().__init__(vectordb_name, vectordb_path, embedding_model)
        encoder_name = embedding_model.model_name_or_path

        class MyEmbeddingFunction(EmbeddingFunction):
            def __call__(self, input: Documents) -> Embeddings:
                from sentence_transformers import SentenceTransformer
                sentences = input
                model = SentenceTransformer(encoder_name)
                embeddings = model.encode(sentences)
                # Convert embeddings to a list of lists
                embeddings_as_list = [embedding.tolist()
                                      for embedding in embeddings]
                return embeddings_as_list

        custom_embeddings = MyEmbeddingFunction()
        del embedding_model
        self.client = chromadb.PersistentClient(path=self.vectordb_path)

        collections = self.client.list_collections()
        if not collections:
            raise ValueError(f"the vectore database has no collections")
        self.collection = self.client.get_collection(
            name=collections[0].name, embedding_function=custom_embeddings)

    def get_similiar_results_cossim(self, q: str, k: int = 3) -> list[tuple[str, float, dict]]:
        results = self.collection.query(query_texts=[q], n_results=k)

        if results['documents'] and results['distances'] and results['metadatas']:
            results_texts = [str(text) for text in results['documents'][0]]
            results_distances = [float(distance)
                                 for distance in results['distances'][0]]
            results_metadata = [dict(metadata) for metadata in results['metadatas'][0]] if results['metadatas'][0][0] else [
                {} for _ in range(len(results['metadatas'][0]))]

            results = [(text, metadata, distance) for text, metadata, distance in zip(
                results_texts, results_distances, results_metadata)]
            return results
        else:
            return []

    def get_all(self) -> list[tuple[str, dict[str, str | int | float | bool]]]:

        results = self.collection.get()
        if results['documents'] and results['metadatas']:
            final_results = []
            metadatas = [dict(metadata) if metadata else {}
                         for metadata in results['metadatas']]
            final_results = [(str(doc), metadata) for doc,
                             metadata in zip(results['documents'], metadatas)]
            return final_results
        else:
            return []

    def get_random(self) -> str:

        all = self.get_all()
        all_documents = [element[0] for element in all]
        return random.choice(all_documents)


SUPPORTED_VECTOR_DATABASES: dict[str, Type[VectorDB]] = {
    'chroma': Chroma
}
