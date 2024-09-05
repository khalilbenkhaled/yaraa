from typing import Tuple
import yaml

from yaraa.RAG.RAG import Basic_RAG
from yaraa.evaluation.metrics import SUPPORTED_METRICS, Evaluator
from yaraa.evaluation.synthetic_data_generator import Synthetic_data_generator
from yaraa.models.base import Generator

SUPPORTED_INFERENCE = {
    'frontend': ['streamlit'],
    'backend': ['fastapi'],
}


def is_valid_keys(data: dict) -> None:
    # general checks
    # check1
    if not set(['Vectorstore', 'Generator', 'Encoder']).issubset(set(data.keys())):
        raise ValueError(
            'make sure the Vectorstore, Generator and the Encoder are passed in.')
    # check2
    if not ('Inference' in data.keys() or 'Evaluation' in data.keys()):
        raise ValueError('you must pass in: either Inference or Evaluation.')
    # check3
    if set(['Inference', 'Evaluation']).issubset(set(data.keys())):
        raise ValueError(
            "you cannot do inference and evaluation at the same time.")
    # vectorstore checks
    if not set(['name', 'path']).issubset(set(data['Vectorstore'].keys())):
        raise ValueError(
            'you need to pass in the name and the pass of the vectorstore')
    # encoder checks
    if not 'model_name' in data['Encoder'].keys():
        raise ValueError('you need to pass in the model_name in the Encoder')
    # generator checks
    if not set(['model_name', 'is_ollama', 'is_server', 'file_name']).issubset(set(data['Generator'].keys())):
        raise ValueError(
            'you need to pass in: model_name, is_ollama, is_server, file_name in the Generator')
    # inference checks
    if 'Inference' in data.keys() and (not ('backend' in data['Inference'].keys() and 'frontend' in data['Inference'].keys())):
        raise ValueError(
            'you need to pass in: backend, frontend in the Inference. Check the docs for the supported inference libraries and frameworks')
    # evaluation checks
    if 'Evaluation' in data.keys() and (not ('metrics' in data['Evaluation'].keys() and 'judge' in data['Evaluation'].keys())):
        raise ValueError(
            'you need to pass in: metrics, judge in the Evaluation')
    # metrics checks in the evaluation
    # if 'Evaluation' in data.keys() and (not set(SUPPORTED_METRICS.keys()).issubset(set(data['Evaluation']['metrics']))):
    if 'Evaluation' in data.keys() and (not set(data['Evaluation']['metrics']).issubset(set(SUPPORTED_METRICS.keys()))):
        raise ValueError(
            f'make sure the metrics you entered are supported. supported metrics: {list(SUPPORTED_METRICS.keys())}')
    # judge checks in the evaluation
    if 'Evaluation' in data.keys() and (not set(['model_name', 'is_ollama', 'is_server', 'file_name']).issubset(set(data['Evaluation']['judge'].keys()))):
        raise ValueError(
            'you need to pass in: model_name, is_ollama, is_server, file_name in the Generator')


def is_valid_values(data: dict) -> None:
    # inference checks
    if 'Inference' in data.keys() and (not (set([data['Inference']['frontend']]).issubset(set(SUPPORTED_INFERENCE['frontend'])))):
        raise ValueError(
            f'the frontend you entered is not currently supported, you must choose from these: {SUPPORTED_INFERENCE["frontend"]}')

    elif 'Inference' in data.keys() and (not (set([data['Inference']['backend']]).issubset(set(SUPPORTED_INFERENCE['backend'])))):
        raise ValueError(
            f'the backend you entered is not currently supported, you must choose from these: {SUPPORTED_INFERENCE["backend"]}')


def parse_yaml(file_name: str) -> Tuple[Basic_RAG, dict, dict]:
    data = {}
    with open(file_name) as stream:
        try:
            data = yaml.safe_load(stream)
        except yaml.YAMLError:
            raise ValueError(f"there was an error parsing the yaml file")
    # checks
    is_valid_keys(data)
    is_valid_values(data)
    rag = Basic_RAG(
        vectordb=data['Vectorstore']['name'],
        vectordb_path=data['Vectorstore']['path'],
        encoder_name_or_path=data['Encoder']['model_name'],
        generator_name_or_path=data['Generator']['model_name'],
        is_ollama=data['Generator']['is_ollama'],
        is_server=data['Generator']['is_server'],
        file_name=data['Generator']['file_name'],
        prompt=data['Generator']['prompt']
    )

    inference = {}
    if 'Inference' in data.keys():
        inference = {
            'frontend': data['Inference']['frontend'],
            'backend': data['Inference']['backend']
        }

    evaluation = {}
    if 'Evaluation' in data.keys():
        synthetic_data_generator = Synthetic_data_generator(
            vectordb=data['Vectorstore']['name'],
            vectordb_path=data['Vectorstore']['path'],
            encoder_name_or_path=data['Encoder']['model_name'],
            generator_name_or_path=data['Generator']['model_name'],
            is_ollama=data['Generator']['is_ollama'],
            is_server=data['Generator']['is_server'],
            file_name=data['Generator']['file_name']
        )
        judge = Generator(
            model_name_or_path=data['Evaluation']['judge']['model_name'],
            is_ollama=data['Evaluation']['judge']['is_ollama'],
            is_server=data['Evaluation']['judge']['is_server'],
            file_name=data['Evaluation']['judge']['file_name']
        )
        evaluator = Evaluator(['answer relevance'], rag, judge)

        evaluation = {
            'synthetic_data_generator': synthetic_data_generator,
            'evaluator': evaluator
        }

    return (rag, inference, evaluation)
