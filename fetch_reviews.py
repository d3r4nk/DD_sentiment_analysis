import json
import os
from pathlib import Path

import pandas as pd
import steamreviews

APP_ID = 262060
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


def flatten_downloaded_json(json_path: Path) -> pd.DataFrame:
    df = pd.read_json(json_path)
    df = df[df["reviews"].apply(lambda x: isinstance(x, dict))].copy()
    expanded = pd.json_normalize(df["reviews"])
    result = pd.concat([df.drop(columns=["reviews"]), expanded], axis=1)
    return result


def main() -> None:
    request_params = {
        "language": "english",
        "review_type": "all",
        "purchase_type": "all",
        "filter": "updated",
    }

    steamreviews.download_reviews_for_app_id_batch([APP_ID], chosen_request_params=request_params)
    print('Downloaded review files via steamreviews. Searching for review JSON...')

    # steamreviews usually writes JSON files to ./data by default.
    candidate_dirs = [PROJECT_ROOT / "data", Path.cwd() / "data"]
    json_files = []
    for d in candidate_dirs:
        if d.exists():
            json_files.extend(sorted(d.glob(f"review_{APP_ID}*.json")))
            json_files.extend(sorted(d.glob(f"*{APP_ID}*.json")))

    if not json_files:
        raise FileNotFoundError(
            "No downloaded review JSON files were found. Check steamreviews output directory."
        )

    frames = []
    for path in json_files:
        try:
            frames.append(flatten_downloaded_json(path))
        except Exception as exc:
            print(f"Skipping {path.name}: {exc}")

    if not frames:
        raise RuntimeError("No review JSON files could be parsed into tabular form.")

    final_df = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["recommendationid"], keep="first")
    out_csv = RAW_DIR / "darkest_dungeon_reviews.csv"
    final_df.to_csv(out_csv, index=False)
    print(f"Saved {len(final_df):,} reviews to {out_csv}")


if __name__ == "__main__":
    main()
