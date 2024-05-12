from engine.base_llm import BaseLLM
from service import move_service


def self_improvement_scoring(move_thoughts: dict, llm: BaseLLM, model_name: str, feature_flags: str):
    # Have an LLM score the thoughts, and generate improvements.
    response = move_service.score_thoughts(llm, model_name, move_thoughts)
    if response['score'] < 4:
        response = self_improvement_scoring(response['improved_response'], llm, model_name)
    return response


def self_consistent_scoring(llm: BaseLLM, model_name: str, feature_flags, n=3):
    response, average_score = None, 0

    while average_score < 4:
        # Generate a chess move with commentary/thoughts.
        response = move_service.get_llm_move(llm, model_name, feature_flags)
        # Have 5 LLMs score the thoughts, and take the average.
        responses = [move_service.score_thoughts(llm, model_name, response) for _ in range(n)]
        average_score = sum([response['score'] for response in responses]) / n

    return response


def self_consistent_improvement(move_thoughts: dict, llm: BaseLLM, model_name: str, n=3):
    # Have 3 LLMs score the thoughts, and generate improvements.
    responses = [move_service.score_thoughts(llm, model_name, move_thoughts) for _ in range(n)]
    # Average the score and aggregate the improved thoughts to include only the best.
    average_score = sum([response['score'] for response in responses]) / n
    improved_response = move_service.aggregate_improvements(responses)
    # While the average score is below 4, keep improving.
    if average_score < 4:
        improved_response = self_consistent_improvement(improved_response, llm, model_name)
    return improved_response


def score():
    pass
