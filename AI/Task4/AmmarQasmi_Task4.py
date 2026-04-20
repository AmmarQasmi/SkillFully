"""Text summarization using a local Ollama model.

This script accepts input text and summarizes it in a short and clear way.
"""

from __future__ import annotations

import sys
import json
from dataclasses import dataclass
from typing import Final
import ollama


SYSTEM_PROMPT: Final[str] = (
	"You are a summarization tool. Summarize text in a short and clear way, making sure "
	"everything is present. Keep the tone professional and formal"
)

DEFAULT_MODEL: Final[str] = "llama3.2:1b"


@dataclass(frozen=True)
class SummarizationResult:
	summary: str
	raw_text: str


def summarize_text(text: str, model: str = DEFAULT_MODEL) -> SummarizationResult:
	"""Return a summary of the given text."""

	response = ollama.chat(
		model=model,
		messages=[
			{"role": "system", "content": SYSTEM_PROMPT},
			{
				"role": "user",
				"content": f"Please summarize the following text:\n\n{text.strip()}",
			},
		],
	)

	raw_text = response["message"]["content"].strip()
	return SummarizationResult(
		summary=raw_text,
		raw_text=raw_text,
	)


def main() -> None:
	"""Read input text and print JSON summarization output."""

	if len(sys.argv) > 1:
		text = " ".join(sys.argv[1:]).strip()
	else:
		text = input("Enter text to summarize: ").strip()

	if not text:
		print(
			json.dumps(
				{
					"input": text,
					"summary": "",
				},
				ensure_ascii=True,
			)
		)
		return

	try:
		result = summarize_text(text)
		print()  # Line spacing between input and output
		print(
			json.dumps(
				{
					"summary": result.summary,
				},
				ensure_ascii=True,
			)
		)
	except Exception as exc:  # pragma: no cover - runtime fallback
		print(f"Error: {exc}", file=sys.stderr)
		sys.exit(1)


if __name__ == "__main__":
	main()
