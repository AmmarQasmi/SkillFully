"""Text classification using a local Ollama model.

This script accepts input text and classifies it into one of the supported
categories.
"""

from __future__ import annotations

import sys
import json
from dataclasses import dataclass
from typing import Final

import ollama


SYSTEM_PROMPT: Final[str] = (
	"You are a text classification tool. Return only valid JSON with no extra text. "
	"Classify input into one or more categories from this list: Sports, Technology, Business, "
	"Entertainment, Health, Education, Crypto, Stocks, Politics. "
	"Output schema: {\"predictions\":[{\"category\":\"Technology\",\"confidence\":80}]}. "
	"Confidence must be an integer from 0 to 100."
)

DEFAULT_MODEL: Final[str] = "llama3.2:1b"

ALLOWED_CATEGORIES: Final[tuple[str, ...]] = (
	"Sports",
	"Technology",
	"Business",
	"Entertainment",
	"Health",
	"Education",
	"Crypto",
	"Stocks",
	"Politics",
)

CATEGORY_MAP: Final[dict[str, str]] = {
	category.lower(): category for category in ALLOWED_CATEGORIES
}


@dataclass(frozen=True)
class ClassificationResult:
	predictions: list[dict[str, int | str]]
	primary_label: str
	raw_text: str


def _normalize_category(value: str) -> str | None:
	"""Normalize model category text to one allowed label."""

	cleaned = value.strip().lower()
	if not cleaned:
		return None

	if cleaned in CATEGORY_MAP:
		return CATEGORY_MAP[cleaned]

	for key, label in CATEGORY_MAP.items():
		if key in cleaned:
			return label

	return None


def _normalize_confidence(value: object) -> int:
	"""Coerce confidence into an integer percentage [0, 100]."""

	if isinstance(value, bool):
		return 0
	if isinstance(value, (int, float)):
		parsed = int(round(value))
	else:
		text = str(value).strip().replace("%", "")
		parsed = int(float(text)) if text else 0

	return max(0, min(100, parsed))


def _fallback_predictions(raw_text: str) -> list[dict[str, int | str]]:
	"""Fallback parser when model output is not valid JSON."""

	cleaned = raw_text.strip().lower()
	predictions: list[dict[str, int | str]] = []
	for category in ALLOWED_CATEGORIES:
		if category.lower() in cleaned:
			predictions.append({"category": category, "confidence": 60})

	if not predictions:
		return [{"category": "Unknown", "confidence": 0}]

	return predictions[:3]


def _build_predictions(raw_text: str) -> list[dict[str, int | str]]:
	"""Build normalized multi-label predictions from model output."""

	try:
		parsed = json.loads(raw_text)
	except json.JSONDecodeError:
		return _fallback_predictions(raw_text)

	items = parsed.get("predictions", []) if isinstance(parsed, dict) else []
	if not isinstance(items, list):
		return _fallback_predictions(raw_text)

	seen: set[str] = set()
	predictions: list[dict[str, int | str]] = []
	for item in items:
		if not isinstance(item, dict):
			continue

		category_raw = str(item.get("category", ""))
		category = _normalize_category(category_raw)
		if not category or category in seen:
			continue

		confidence = _normalize_confidence(item.get("confidence", 0))
		predictions.append({"category": category, "confidence": confidence})
		seen.add(category)

	if not predictions:
		return _fallback_predictions(raw_text)

	predictions.sort(key=lambda item: int(item["confidence"]), reverse=True)
	return predictions[:3]


def classify_text(text: str, model: str = DEFAULT_MODEL) -> ClassificationResult:
	"""Return normalized multi-label classifications for the given text."""

	response = ollama.chat(
		model=model,
		messages=[
			{"role": "system", "content": SYSTEM_PROMPT},
			{
				"role": "user",
				"content": (
					"Classify the following text and return JSON only using the required schema.\n"
					f"{text.strip()}"
				),
			},
		],
	)

	raw_text = response["message"]["content"].strip()
	predictions = _build_predictions(raw_text)
	primary_label = str(predictions[0]["category"]) if predictions else "Unknown"
	return ClassificationResult(
		predictions=predictions,
		primary_label=primary_label,
		raw_text=raw_text,
	)


def main() -> None:
	"""Read input text and print JSON classification output."""

	if len(sys.argv) > 1:
		text = " ".join(sys.argv[1:]).strip()
	else:
		text = input("Enter text to classify: ").strip()

	if not text:
		print(
			json.dumps(
				{
					"input": text,
					"primary_label": "Unknown",
					"predictions": [],
				},
				ensure_ascii=True,
			)
		)
		return

	try:
		result = classify_text(text)
		print(
			json.dumps(
				{
					"input": text,
					"primary_label": result.primary_label,
					"predictions": result.predictions,
				},
				ensure_ascii=True,
			)
		)
	except Exception as exc:  # pragma: no cover - runtime fallback
		print(f"Error: {exc}", file=sys.stderr)
		sys.exit(1)


if __name__ == "__main__":
	main()
