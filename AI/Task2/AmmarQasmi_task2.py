"""Sentiment analysis using a local Ollama model.

This script accepts a sentence, review, or crypto-news headline and asks a
local Ollama model to classify it as Positive, Negative, or Neutral.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Final

import ollama


SYSTEM_PROMPT: Final[str] = (
	"You are a sentiment analysis tool. Only reply with Positive, Negative, or Neutral."
)

DEFAULT_MODEL: Final[str] = "llama3.2:1b"


@dataclass(frozen=True)
class SentimentResult:
	label: str
	raw_text: str


def _normalize_label(content: str) -> str:
	"""Extract a valid sentiment label from model output."""

	cleaned = content.strip().lower()
	if cleaned.startswith("positive"):
		return "Positive"
	if cleaned.startswith("negative"):
		return "Negative"
	return "Neutral"


def classify_sentiment(text: str, model: str = DEFAULT_MODEL) -> SentimentResult:
	"""Return a single sentiment label for the given text."""

	response = ollama.chat(
		model=model,
		messages=[
			{"role": "system", "content": SYSTEM_PROMPT},
			{
				"role": "user",
				"content": (
					"Classify the sentiment of this crypto/news text:\n"
					f"{text.strip()}"
				),
			},
		],
	)

	raw_text = response["message"]["content"].strip()
	return SentimentResult(label=_normalize_label(raw_text), raw_text=raw_text)


def main() -> None:
	"""Read input text and print the predicted sentiment label."""

	if len(sys.argv) > 1:
		text = " ".join(sys.argv[1:]).strip()
	else:
		text = input("Enter a sentence, review, or crypto news text: ").strip()

	if not text:
		print("Neutral")
		return

	try:
		print(classify_sentiment(text).label)
	except Exception as exc:  # pragma: no cover - runtime fallback
		print(f"Error: {exc}", file=sys.stderr)
		sys.exit(1)


if __name__ == "__main__":
	main()
