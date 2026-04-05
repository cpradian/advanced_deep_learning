from .base_llm import BaseLLM


class CoTModel(BaseLLM):
    def format_prompt(self, question: str) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a careful unit conversion assistant. "
                    "Solve the problem briefly and correctly. "
                    "Only output a short reasoning chain followed by the final answer. "
                    "Always end with the final numeric answer inside <answer></answer>. "
                    "Do not put units inside the answer tag. "
                    "Do not ask follow-up questions. "
                    "Do not output multiple choice options."
                ),
            },
            {
                "role": "user",
                "content": "How many gram are there in 2 kg?",
            },
            {
                "role": "assistant",
                "content": (
                    "1 kg = 1000 gram. "
                    "2 kg = 2 * 1000 = 2000. "
                    "<answer>2000</answer>"
                ),
            },
            {
                "role": "user",
                "content": "How many feet are there in 3 yard?",
            },
            {
                "role": "assistant",
                "content": (
                    "1 yard = 3 feet. "
                    "3 yard = 3 * 3 = 9. "
                    "<answer>9</answer>"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"{question}\n"
                    "Respond with brief reasoning and then the final numeric answer in <answer></answer>."
                ),
            },
        ]

        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def load() -> CoTModel:
    return CoTModel()


def test_model():
    from .data import Dataset, benchmark

    testset = Dataset("valid")
    model = CoTModel()
    benchmark_result = benchmark(model, testset, 100)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")


if __name__ == "__main__":
    from fire import Fire

    Fire({"test": test_model, "load": load})