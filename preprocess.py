import re
from pathlib import Path

import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "darkest_dungeon_reviews.csv"
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "darkest_dungeon_reviews_clean.csv"


nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

def clean_reviews(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["review"]).copy()
    if "language" in df.columns:
        df = df[df["language"].astype(str).str.lower() == "english"].copy()

    mask = df["review"].apply(
        lambda x: len(re.findall(r"\b\w+\b", str(x))) > 5 and bool(re.search(r"[a-zA-Z]", str(x)))
    )
    df = df[mask].copy()

    df["review_length"] = df["review"].astype(str).str.split().str.len()

    if "author.playtime_forever" in df.columns:
        df["playtime_hours"] = df["author.playtime_forever"].fillna(0) / 60.0
        q1, q2, q3 = df["playtime_hours"].quantile([0.25, 0.50, 0.75])
        df["playtime_bucket"] = pd.cut(
            df["playtime_hours"],
            bins=[-np.inf, q1, q2, q3, np.inf],
            labels=["low", "medium", "high", "very_high"],
        )

    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()

    def lemmatize_text(text: str) -> str:
        words = word_tokenize(str(text).lower())
        words = [re.sub(r"[^a-z]", "", w) for w in words]
        words = [w for w in words if w and w not in stop_words]
        words = [lemmatizer.lemmatize(w) for w in words]
        return " ".join(words)

    df["review_clean"] = df["review"].apply(lemmatize_text)
    return df


def main() -> None:
    df = pd.read_csv(RAW_PATH)
    clean_df = clean_reviews(df)
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(PROCESSED_PATH, index=False)
    print(f"Saved cleaned dataset to {PROCESSED_PATH}")
    print(clean_df[["review", "review_clean"]].head())


if __name__ == "__main__":
    main()
