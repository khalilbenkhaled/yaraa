_simple = """
based on the following information delimited by a ### generate a question where the answer to the question is somewhere in the provided information,
you should not mention anything about the information in the question.
give your answer directly no introductions.
information: ### {context} ###
question:
"""

_multiple = """
based on the following information delimited by a ### generate a question where the answer to the question is somewhere in the provided information,
you should not mention anything about the information in the question.
give your answer directly no introductions.
information: ### {context} ###
question:
"""

_reasoning = """
based on the following information delimited by a ### generate a question where the answer to the question is somewhere in the provided information,
you should not mention anything about the information in the question.
give your answer directly no introductions.
information: ### {context} ###
question:
"""

PROMPTS_SYNTHETIC_DATA_GENERATION = {
    'simple': _simple,
    'multiple': _multiple,
    'reasoning': _reasoning
}

_answer_relevance = """
generate a question from the following context delimited by a ((( ))), where the answer to the question should an information presented in the context.
context: (((
{context}
)))
"""

PROMPTS_EVALUATION = {
    'answer relevance': _answer_relevance
}
