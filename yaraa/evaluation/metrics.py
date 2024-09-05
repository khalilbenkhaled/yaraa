from abc import ABC, abstractmethod
import numpy as np
from numpy.linalg import norm

from typing import Optional, Type, Union


from yaraa.RAG.RAG import Basic_RAG
from yaraa.models.base import Encoder, Generator
from yaraa.evaluation.prompts import PROMPTS_EVALUATION


DEFAULT_ENCODER = Encoder('sentence-transformers/all-mpnet-base-v2')


class Metric(ABC):
    def __init__(self, data: Union[list[str], list[dict]], rag: Optional[Basic_RAG] = None, judge: Optional[Generator] = None) -> None:
        self.name = ''
        self.data = data
        self.rag = rag
        self.judge = judge

        if all(isinstance(item, str) for item in self.data):
            if not rag:
                raise ValueError('you should pass in the rag pipeline')
        else:
            if not set(['answer', 'context', 'question']).issubset(set(data[0])):
                raise ValueError(
                    'the data dict should have the following keys: question, context and answer')

    @abstractmethod
    def _evaluate_with_answers(self, judge: Generator, n=3) -> float:
        raise NotImplementedError

    @abstractmethod
    def _evaluate_without_answers(self, rag: Basic_RAG, judge: Generator, n=3) -> float:
        raise NotImplementedError

    @abstractmethod
    def evaluate(self,  judge: Generator, rag: Optional[Basic_RAG] = None, n=3) -> float:
        raise NotImplementedError


class Answer_relevance(Metric):
    def __init__(self, data: Union[list, dict], rag: Optional[Basic_RAG] = None, judge: Optional[Generator] = None) -> None:
        super().__init__(data, rag, judge)
        self.name = 'answer relevance'

    def _calculate_cossin(self, A: np.ndarray, B: np.ndarray) -> float:
        return np.dot(A, B)/(norm(A)*norm(B))

    def _calculate_cossim_metric(self, true_question: str, questions: list[str]) -> float:
        """
        calculates and return the average cosine sim between the desired question and other generated questions
        """
        e_true_question = DEFAULT_ENCODER.encode(true_question)
        scores = []
        for question in questions:
            e_question = DEFAULT_ENCODER.encode(question)
            score = self._calculate_cossin(e_true_question, e_question)
            scores.append(score)
        return float(np.average(scores))

    def _evaluate_with_answers(self, judge: Generator, n=3) -> float:

        scores_final = []
        for element in self.data:
            context_strings = "".join(element['context'])
            prompt = PROMPTS_EVALUATION[self.name].format(
                context=context_strings)

            questions: list = []
            for _ in range(n):
                question = judge.generate(prompt)
                questions.append(question)
            score = self._calculate_cossim_metric(
                element['question'], questions)
            scores_final.append(score)
        final_score = float(np.average(scores_final))
        return final_score

    def _evaluate_without_answers(self, rag: Basic_RAG, judge: Generator, n=3) -> float:
        data_final: list[dict[str, str | list[str]]] = []
        for question in self.data:
            sample: dict[str, str | list[str]] = {}
            answer, context = rag.generate(question)
            context = [element[0] for element in context]
            sample['question'] = question
            sample['answer'] = answer
            sample['context'] = context
            data_final.append(sample)
        self.data = data_final

        score = self._evaluate_with_answers(judge)

        return score

    def evaluate(self,  judge: Generator, rag: Optional[Basic_RAG] = None, n=3) -> float:
        if all(isinstance(item, str) for item in self.data):
            return self._evaluate_without_answers(rag, judge)
        else:
            return self._evaluate_with_answers(judge)


class Context_precision(Metric):
    pass


SUPPORTED_METRICS: dict[str, Type[Metric]] = {
    'answer relevance': Answer_relevance,
    'context precision': Context_precision
}


class Evaluator:
    def __init__(self, metrics: list[str], rag: Basic_RAG, judge: Generator) -> None:
        self.rag = rag
        self.judge = judge
        self.metrics = metrics

    def evaluate(self, data) -> dict: 
        results: dict[str, float] = {}
        self.metrics_instances = [SUPPORTED_METRICS[element](
            data, self.rag, self.judge) for element in self.metrics]
        for metric in self.metrics_instances:
            results[metric.name] = metric.evaluate(self.judge, self.rag)
        return results
