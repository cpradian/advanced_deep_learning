import json
from pathlib import Path

from .cot import CoTModel
from .data import Dataset, is_answer_valid


def generate_dataset(output_json: str, oversample: int = 10, temperature: float = 0.6):
    trainset = Dataset("train")
    model = CoTModel()

    questions = [item[0] for item in trainset]
    correct_answers = [item[1] for item in trainset]

    generations = model.batched_generate(
        questions,
        num_return_sequences=oversample,
        temperature=temperature,
    )

    output_data = []
    n_success = 0

    for question, correct_answer, candidates in zip(questions, correct_answers, generations):
        chosen_reasoning = None

        for candidate in candidates:
            parsed = model.parse_answer(candidate)
            if is_answer_valid(parsed, correct_answer):
                chosen_reasoning = candidate.strip()
                break

        if chosen_reasoning is not None:
            output_data.append([question, float(correct_answer), chosen_reasoning])
            n_success += 1

    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w") as f:
        json.dump(output_data, f, indent=2)

    print(f"Saved {len(output_data)} / {len(trainset)} successful samples to {output_path}")
    print(f"Success rate: {len(output_data) / len(trainset):.3f}")


if __name__ == "__main__":
    from fire import Fire

    Fire(generate_dataset)