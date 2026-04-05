import json
import re
from pathlib import Path

from .cot import CoTModel
from .data import Dataset, is_answer_valid


def extract_numbers(text: str) -> list[float]:
    """
    Extract candidate numeric values from free-form text.
    Handles integers, decimals, negatives, and simple scientific notation.
    """
    text = text.replace(",", "")
    matches = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    numbers = []
    for m in matches:
        try:
            numbers.append(float(m))
        except ValueError:
            pass
    return numbers


def generate_dataset(output_json: str, oversample: int = 10, temperature: float = 0.5):
    trainset = Dataset("train")

    # Larger instruct model helps rollout quality
    model = CoTModel(checkpoint="HuggingFaceTB/SmolLM2-1.7B-Instruct")

    questions = [item[0] for item in trainset]
    correct_answers = [item[1] for item in trainset]

    generations = model.batched_generate(
        questions,
        num_return_sequences=oversample,
        temperature=temperature,
    )

    output_data = []
    n_direct_tag = 0
    n_regex_salvaged = 0

    for question, correct_answer, candidates in zip(questions, correct_answers, generations):
        chosen_reasoning = None

        for candidate in candidates:
            cleaned = candidate.replace(",", "").strip()

            # 1) Try normal tag-based parse first
            parsed = model.parse_answer(cleaned)
            if is_answer_valid(parsed, correct_answer):
                chosen_reasoning = candidate.strip()
                n_direct_tag += 1
                break

            # 2) Fall back to scanning all numbers in the text
            nums = extract_numbers(candidate)
            valid_nums = [x for x in nums if is_answer_valid(x, correct_answer)]

            if valid_nums:
                best = valid_nums[-1]  # often the final number is the answer
                reasoning = candidate.strip()

                # Append canonical answer tag if missing
                if "<answer>" not in reasoning:
                    reasoning = f"{reasoning} <answer>{round(best, 6)}</answer>"

                chosen_reasoning = reasoning
                n_regex_salvaged += 1
                break

        if chosen_reasoning is not None:
            output_data.append([question, float(correct_answer), chosen_reasoning])

    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w") as f:
        json.dump(output_data, f, indent=2)

    print(f"Saved {len(output_data)} / {len(trainset)} successful samples to {output_path}")
    print(f"Success rate: {len(output_data) / len(trainset):.3f}")
    print(f"Direct tag matches: {n_direct_tag}")
    print(f"Regex-salvaged matches: {n_regex_salvaged}")


if __name__ == "__main__":
    from fire import Fire

    Fire(generate_dataset)