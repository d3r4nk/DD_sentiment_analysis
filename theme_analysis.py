import json
import re
from pathlib import Path
from typing import Dict, List

import pandas as pd
from textblob import TextBlob

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "darkest_dungeon_reviews_clean.csv"
THEMES_PATH = PROJECT_ROOT / "config" / "themes.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "darkest_dungeon_reviews_with_theme_scores.csv"


def load_themes() -> Dict[str, List[str]]:
    with open(THEMES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def sentence_mentions_theme(sentence: str, keywords: List[str]) -> bool:
    sentence = sentence.lower()
    for kw in keywords:
        pattern = r"\b" + re.escape(kw.lower()) + r"\b"
        if re.search(pattern, sentence):
            return True
    return False


def score_themes(review: str, themes: Dict[str, List[str]]) -> Dict[str, float]:
    review = str(review)
    blob = TextBlob(review)
    result = {}
    result["overall_polarity"] = blob.sentiment.polarity
    result["overall_subjectivity"] = blob.sentiment.subjectivity

    sentences = [str(s) for s in blob.sentences] or [review]
    for theme, keywords in themes.items():
        matching_scores = []
        mention_count = 0
        for sentence in sentences:
            if sentence_mentions_theme(sentence, keywords):
                mention_count += 1
                matching_scores.append(TextBlob(sentence).sentiment.polarity)
        result[f"{theme}_mentions"] = mention_count
        result[f"{theme}_polarity"] = sum(matching_scores) / len(matching_scores) if matching_scores else 0.0
    return result


def main() -> None:
    df = pd.read_csv(PROCESSED_PATH)
    themes = load_themes()
    theme_df = df["review"].apply(lambda x: pd.Series(score_themes(x, themes)))
    final_df = pd.concat([df, theme_df], axis=1)
    final_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved theme-scored dataset to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
