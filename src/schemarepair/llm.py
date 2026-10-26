"""LLM access for schemarepair.

The benchmark and the agent talk to any object implementing ``LLMClient``:
``complete(messages, seed=..., max_new_tokens=..., temperature=...)`` ->
``LLMResponse``. ``LocalLLM`` runs a small open-weight model on CPU via
transformers; the test suite uses scripted fakes. Token counts always come
from the model's own tokenizer, never estimated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class LLMResponse:
    text: str
    prompt_tokens: int
    completion_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class LLMClient(Protocol):
    def complete(self, messages: list[dict[str, str]], *, seed: int = 0,
                 max_new_tokens: int = 192, temperature: float = 0.8) -> LLMResponse:
        ...


class LocalLLM:
    """A local open-weight chat model on CPU (transformers backend)."""

    def __init__(self, model_dir: str, *, threads: int = 2):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch.set_num_threads(threads)
        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir, dtype=torch.float32)
        self.model.eval()

    def complete(self, messages: list[dict[str, str]], *, seed: int = 0,
                 max_new_tokens: int = 192, temperature: float = 0.8) -> LLMResponse:
        torch = self._torch
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        prompt_tokens = int(inputs.input_ids.shape[1])
        # transformers 5.x generate() no longer accepts a per-call generator;
        # seed the global RNG (generation here is single-stream, so this is
        # deterministic for a given seed).
        torch.manual_seed(seed)
        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=True,
                temperature=temperature, top_p=0.95, repetition_penalty=1.08,
                pad_token_id=self.tokenizer.eos_token_id)
        new = out[0][prompt_tokens:]
        text = self.tokenizer.decode(new, skip_special_tokens=True)
        return LLMResponse(text=text, prompt_tokens=prompt_tokens,
                           completion_tokens=int(new.shape[0]))
