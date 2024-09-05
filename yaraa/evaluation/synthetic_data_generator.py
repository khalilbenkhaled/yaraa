from yaraa.models.base import Encoder, Generator
from yaraa.models.vectordb import SUPPORTED_VECTOR_DATABASES
from yaraa.evaluation.prompts import PROMPTS_SYNTHETIC_DATA_GENERATION
DEFAULT_DISTRIBUTION = {
    'simple': 0.5,
    'multiple': 0.25,
    'reasoning': 0.25,
}


class Synthetic_data_generator:
    def handle_general_instantiation(self, kwargs):

        self.encoder = Encoder(kwargs['encoder_name_or_path'])

        generator_name_or_path = kwargs['generator_name_or_path']

        is_ollama = kwargs['is_ollama'] if 'is_ollama' in kwargs.keys(
        ) else False
        is_server = kwargs['is_server'] if 'is_server' in kwargs.keys(
        ) else False
        file_name = kwargs['file_name'] if 'file_name' in kwargs.keys() else ''
        self.generator = Generator(
            generator_name_or_path, is_ollama, is_server, file_name)
        self.vectordb = SUPPORTED_VECTOR_DATABASES[kwargs['vectordb']](
            vectordb_name=kwargs['vectordb'], vectordb_path=kwargs['vectordb_path'], embedding_model=self.encoder)

    def handle_particular_instantiation(self, kwargs):
        self.encoder = kwargs['encoder']
        self.generator = kwargs['generator']
        self.vectordb = kwargs['vectordb']

    def __init__(self, **kwargs) -> None:
        if set(['vectordb', 'vectordb_path', 'encoder_name_or_path', 'generator_name_or_path']).issubset(list(kwargs.keys())):
            self.handle_general_instantiation(kwargs)

        elif set(['generator', 'encoder', 'vectordb']).issubset(list(kwargs.keys())):
            self.handle_particular_instantiation(kwargs)
        else:
            raise ValueError(
                "you must either specify ['vectordb', 'vectordb_path', 'encoder_name_or_path', 'generator_name_or_path', 'is_ollama', 'is_server', 'file_name'] or ['generator', 'encoder', 'vectordb']")

    def _generate(self, prompt) -> str:

        random_chunk = self.vectordb.get_random()
        chunks_for_generation = self.vectordb.get_similiar_results_cossim(
            random_chunk, k=2)
        chunks_for_generation = [element[0]
                                 for element in chunks_for_generation]
        chunks_for_generation.append(random_chunk)
        results_str = [chunk[0] for chunk in chunks_for_generation]
        chunk_for_generation_str = "".join(results_str)
        prompt = prompt.format(context=chunk_for_generation_str)
        question = self.generator.generate(prompt)
        return question

    # TODO: validate the distribution

    def generate(self, question_number=4, distribution=DEFAULT_DISTRIBUTION) -> list[str]:

        distribution = {key: int(question_number * value)
                        for key, value in distribution.items()}
        questions: list[str] = []
        for key in distribution:
            n = distribution[key]
            for _ in range(n):
                question = self._generate(
                    PROMPTS_SYNTHETIC_DATA_GENERATION[key])
                questions.append(question)
        return questions
