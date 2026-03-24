# Darkest Dungeon Steam Review NLP Analysis

This repository analyzes **Steam reviews for _Darkest Dungeon_** with a focus on:

- text cleaning and normalization
- sentiment analysis
- theme-level sentiment scoring
- topic modeling with LDA
- n-gram exploration
- baseline predictive modeling from review text

The notebook behind this project is:

- `darkest-dungeon-review-sentiment-analysis.ipynb`

---

## Project outputs at a glance

### 1) Preprocessing preview

The cleaned dataset keeps the original review text and creates a normalized text column used by later NLP steps.

![Preprocessing preview](image/preprocess.png)

### 2) Review length distribution

Most reviews are short, with a long right tail of detailed reviews.

![Review length distribution](image/review_length_distribution.png)

### 3) General sentiment score distribution

The sentiment distribution is strongly skewed toward positive values.

![General sentiment score distribution](image/general_sentiment_scores_distribution.png)

### 4) Logistic Regression on `voted_up`

Baseline text classification for whether a review is recommended.

![Logistic Regression on voted_up](image/positive_prediction.png)

### 5) Logistic Regression on `playtime_bucket`

Multi-class baseline that predicts playtime segment from cleaned review text.

![Logistic Regression on playtime bucket](image/playtime_logistic_regression.png)

### 6) XGBoost on above/below-average playtime

Binary prediction of whether a player is above the mean playtime threshold based on review text.

![XGBoost playtime prediction](image/playtime_prediction.png)

### 7) Theme mention counts

Combat and progression/difficulty are the most frequently mentioned themes.

![Theme mentions](image/theme_mention.png)

### 8) LDA topics

Topic modeling reveals clusters around atmosphere/art, turn-based RPG identity, heroes/party/stress, and general evaluation language.

![LDA topics](image/topic_lda.png)

### 9) Top bigrams

Frequent bigrams reflect game identity, genre language, art style, and review intent.

![Top bigrams](image/bigram_count.png)

---

# Data structure

The notebook works with a **processed review dataset** loaded from:

```python
PROCESSED_PATH = PROJECT_ROOT / 'data' / 'processed' / 'darkest_dungeon_reviews_clean.csv'
```

It also expects a raw Steam review export at:

```python
RAW_PATH = PROJECT_ROOT / 'data' / 'raw' / 'darkest_dungeon_reviews.csv'
```

## Core columns used directly by the notebook

The following columns are **verified from the notebook code** and are the fields required for the analysis pipeline.

| Column | Type | Meaning | Where it is used |
|---|---|---|---|
| `review` | text | Original Steam review text written by the user. | Input for cleaning, sentiment analysis, and inspection. |
| `language` | categorical/text | Review language. The notebook keeps only English reviews when this column exists. | Filtering during preprocessing. |
| `voted_up` | boolean or binary-like text | Steam recommendation label. `True` / `1` means recommended, `False` / `0` means not recommended. | Sentiment classification baseline. |
| `author.playtime_forever` | numeric | Total playtime in minutes from the Steam review export. | Converted to hours and used for playtime modeling. |
| `review_length` | integer | Number of whitespace-separated words in the original review. | EDA and random review display. |
| `playtime_hours` | float | Total playtime converted from minutes to hours. | EDA and derived targets. |
| `playtime_bucket` | categorical | Quartile-based label: `low`, `medium`, `high`, `very_high`. | Multi-class Logistic Regression baseline. |
| `review_clean` | text | Cleaned and normalized review text after tokenization, stopword filtering, contraction expansion, and lemmatization. | TF-IDF, LDA, bigrams, and modeling. |

## Engineered columns created during sentiment and theme analysis

These columns are added after scoring each review with **TextBlob + NaiveBayesAnalyzer** and the theme keyword dictionaries from `config/themes.json`.

| Column | Type | Meaning |
|---|---|---|
| `overall_classification` | categorical | Predicted sentiment label from NaiveBayesAnalyzer, typically `pos` or `neg`. |
| `overall_p_pos` | float | Probability that the review is positive. |
| `overall_p_neg` | float | Probability that the review is negative. |
| `general_sentiment_score` | float | Computed as `p_pos - p_neg`, ranging from `-1` to `1`. |
| `combat_mentions` | integer | Number of sentences in the review that mention combat-related keywords. |
| `combat_sentiment_score` | float | Mean sentence-level sentiment score for combat-related sentences. |
| `progression_difficulty_mentions` | integer | Number of sentences that mention progression/difficulty keywords. |
| `progression_difficulty_sentiment_score` | float | Mean sentiment score for progression/difficulty sentences. |
| `art_atmosphere_mentions` | integer | Number of sentences that mention art/atmosphere keywords. |
| `art_atmosphere_sentiment_score` | float | Mean sentiment score for art/atmosphere sentences. |
| `stress_psychology_mentions` | integer | Number of sentences that mention stress/psychology keywords. |
| `stress_psychology_sentiment_score` | float | Mean sentiment score for stress/psychology sentences. |

## Modeling helper columns

These are temporary or modeling-specific columns created inside the notebook.

| Column | Type | Meaning |
|---|---|---|
| `voted_up_binary` | integer | Normalized binary label for `voted_up`, where `1 = recommended`, `0 = not recommended`. |
| `playtime_hours_model` | float | Numeric playtime column prepared for XGBoost. |
| `above_avg_playtime` | integer | Binary target where `1` means playtime is greater than or equal to the mean and `0` means below mean. |

## About additional raw columns

The raw Steam export may include more metadata than the notebook actively uses, depending on the crawler/export format. In this project, the analysis is centered on the verified fields above. If your raw file contains more Steam metadata, those columns are preserved unless explicitly filtered out during preprocessing.

---

# Data preprocessing pipeline

The notebook performs a staged preprocessing flow before any modeling or topic extraction.

## 1) Drop missing reviews

Reviews with missing review text are removed first.

```python
clean_df = df.dropna(subset=['review']).copy()
```

## 2) Keep only English reviews

If the `language` column exists, the notebook filters to English only.

```python
if 'language' in clean_df.columns:
    clean_df = clean_df[
        clean_df['language'].astype(str).str.lower() == 'english'
    ].copy()
```

## 3) Remove very short or invalid reviews

The notebook keeps only reviews that:

- contain more than 5 tokens
- contain at least one alphabetic character

```python
clean_df = clean_df[
    clean_df['review'].apply(
        lambda x: len(re.findall(r'\b\w+\b', str(x))) > 5
        and bool(re.search(r'[a-zA-Z]', str(x)))
    )
].copy()
```

## 4) Compute review length

A simple review length feature is created from the original text.

```python
clean_df['review_length'] = clean_df['review'].astype(str).str.split().str.len()
```

## 5) Convert playtime to hours and create quartile buckets

When the raw Steam field `author.playtime_forever` exists, the notebook:

- converts minutes to hours
- creates a quartile-based target for multi-class prediction

```python
if 'author.playtime_forever' in clean_df.columns:
    clean_df['playtime_hours'] = clean_df['author.playtime_forever'].fillna(0) / 60.0
    clean_df['playtime_bucket'] = pd.qcut(
        clean_df['playtime_hours'].rank(method='first'),
        4,
        labels=['low', 'medium', 'high', 'very_high']
    )
```

## 6) Normalize text before tokenization

The text normalization step handles:

- HTML entities
- Unicode normalization
- line breaks and tabs
- URLs
- HTML tags
- lowercase conversion
- contraction expansion

```python
def normalize_text(text: str) -> str:
    text = html.unescape(str(text))
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = expand_contractions(text.lower())
    return text
```

## 7) Expand contractions

A custom contraction dictionary is used so tokens such as `can't` do not become noisy fragments like `ca` and `nt`.

```python
contractions_map = {
    "can't": "can not",
    "won't": "will not",
    "don't": "do not",
    "isn't": "is not",
    "you're": "you are",
    "they're": "they are",
    "n't": " not",
    "'re": " are",
    "'ve": " have",
    "'ll": " will",
    "'d": " would",
    "'m": " am"
}
```

```python
def expand_contractions(text: str) -> str:
    text = str(text)
    for k, v in contractions_map.items():
        text = re.sub(rf"\b{re.escape(k)}\b", v, text, flags=re.IGNORECASE)
    return text
```

## 8) Tokenize, remove noise, keep negation, and lemmatize

The cleaning function:

- tokenizes text
- strips non-alphabetic characters
- removes one-character tokens
- removes standard English stopwords
- **keeps key negation words** such as `not`, `no`, `never`
- removes custom generic words such as `game`, `player`, `steam`
- lemmatizes tokens
- joins the cleaned tokens back into one string

```python
stop_words = set(stopwords.words('english'))
negation_keep = {"no", "not", "nor", "never"}
custom_remove = {
    "game", "play", "player", "steam",
    "one", "get", "got", "make", "made", "really",
    "thing", "things", "would", "could", "also"
}
final_stopwords = (stop_words - negation_keep) | custom_remove
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = normalize_text(text)
    text = re.sub(r"[^a-z'\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    words = word_tokenize(text)

    cleaned_words = []
    for w in words:
        w = re.sub(r"[^a-z]", "", w)
        if not w:
            continue
        if len(w) <= 1:
            continue
        if w in final_stopwords:
            continue
        lemma = lemmatizer.lemmatize(w)
        if lemma and lemma not in final_stopwords:
            cleaned_words.append(lemma)

    return " ".join(cleaned_words)
```

## 9) Create the final cleaned review text

```python
clean_df['review_clean'] = clean_df['review'].apply(clean_text)
```

## 10) Remove reviews that become too sparse after cleaning

Very weak post-cleaning documents are filtered out.

```python
clean_df = clean_df[
    clean_df['review_clean'].apply(lambda x: len(str(x).split()) >= 3)
].copy()
```

## 11) Save the processed dataset

```python
PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
clean_df.to_csv(PROCESSED_PATH, index=False)
```

---

# Why this preprocessing matters

These preprocessing choices improve more than just LDA:

- **TF-IDF features** become cleaner and less sparse
- **Logistic Regression** sees fewer noisy tokens
- **XGBoost** receives stronger text-derived features
- **theme matching** misses fewer relevant sentences
- **bigrams** become more interpretable
- **topic modeling** becomes less polluted by broken fragments such as `nt`, `ca`, `u`, `b`

The biggest practical improvement came from **contraction handling** and **custom stopword filtering**, which reduced broken tokens and generic filler words in the topic output.

---

# Modeling and analysis steps

## Step 1: Baseline multi-class Logistic Regression on playtime buckets

The notebook uses TF-IDF features from `review_clean` and predicts:

- `low`
- `medium`
- `high`
- `very_high`

```python
clf = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1,2))),
    ('logreg', LogisticRegression(max_iter=2000))
])
```

Observed result from the notebook screenshot:

- accuracy around **0.32**
- strongest class among the four buckets is **low**
- overall result suggests review text alone is only a weak signal for exact playtime quartile

## Step 2: Logistic Regression on `voted_up`

This step predicts whether a review is recommended from the cleaned text.

```python
logreg_sentiment = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
    ('logreg', LogisticRegression(max_iter=2000))
])
```

Observed result from the notebook screenshot:

- label distribution is highly imbalanced toward recommended reviews
- accuracy around **0.93**
- recall for recommended reviews is very high
- minority-class performance is weaker, which is expected given the class imbalance

## Step 3: XGBoost on above/below-average playtime

This step reframes playtime as a binary task around the mean threshold.

```python
xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='binary:logistic',
    eval_metric='logloss',
    random_state=42
)
```

Observed result from the notebook screenshot:

- mean threshold is about **148.05 hours**
- overall accuracy is about **0.71**
- recall is strong for the below-average class
- recall is weak for the above-average class

This indicates that review text contains some signal about playtime intensity, but not enough to cleanly separate high-engagement players from lower-engagement players.

## Step 4: Sentiment analysis with TextBlob NaiveBayesAnalyzer

The notebook computes:

- overall sentiment label
- positive probability
- negative probability
- a general score: `p_pos - p_neg`

```python
nba = NaiveBayesAnalyzer()

def nba_score(text):
    sentiment = TextBlob(text, analyzer=nba).sentiment
    p_pos = float(sentiment.p_pos)
    p_neg = float(sentiment.p_neg)
    return {
        'classification': sentiment.classification,
        'p_pos': p_pos,
        'p_neg': p_neg,
        'score': p_pos - p_neg
    }
```

The sentiment histogram shows that the review distribution is heavily skewed toward positive sentiment.

## Step 5: Theme-based sentence scoring

The notebook loads a manual keyword dictionary from `config/themes.json` and scores sentences that mention each theme.

```python
with open(THEMES_PATH, 'r', encoding='utf-8') as f:
    themes = json.load(f)

def sentence_mentions_theme(sentence, keywords):
    sentence = sentence.lower()
    return any(re.search(r'\b' + re.escape(k.lower()) + r'\b', sentence) for k in keywords)
```

Each theme gets:

- mention count
- mean sentence-level sentiment score

The current theme set includes:

- combat
- progression_difficulty
- art_atmosphere
- stress_psychology

From the notebook outputs, the most frequently mentioned themes are:

1. `combat_mentions`
2. `progression_difficulty_mentions`
3. `art_atmosphere_mentions`
4. `stress_psychology_mentions`

## Step 6: Topic modeling with LDA

The notebook trains a 6-topic LDA model on tokenized `review_clean`.

```python
tokens = [txt.split() for txt in eda_df['review_clean'].dropna().astype(str)]
dictionary = corpora.Dictionary(tokens)
dictionary.filter_extremes(no_below=10, no_above=0.5)
corpus = [dictionary.doc2bow(text) for text in tokens]

lda = LdaModel(
    corpus=corpus,
    id2word=dictionary,
    num_topics=6,
    random_state=42,
    passes=10
)
```

Observed topics include clusters around:

- art style / gameplay / atmosphere / voice
- turn-based dark RPG identity
- heroes / party / characters
- stress / combat / enemies / attacks
- general evaluation language

The LDA output improved after contraction handling, but some topics still contain generic words such as `like`, `good`, `time`, and `not`.

## Step 7: Bigram mining

The notebook extracts 2-grams from cleaned text with `CountVectorizer`.

```python
vectorizer = CountVectorizer(ngram_range=(2,2), max_features=40)
X = vectorizer.fit_transform(eda_df['review_clean'].fillna(''))
counts = np.asarray(X.sum(axis=0)).ravel()
vocab = np.array(vectorizer.get_feature_names_out())
bigrams = pd.DataFrame({'bigram': vocab, 'count': counts}).sort_values('count', ascending=False)
```

Top bigrams shown in the notebook include:

- `darkest dungeon`
- `turn based`
- `art style`
- `dungeon crawler`
- `dark soul`
- `heart attack`
- `party member`
- `not recommend`
- `highly recommend`

These align well with the game’s identity, mechanics, and recommendation language.

---

# Folder assumptions for GitHub

This README expects the following image paths inside the repository:

```text
image/preprocess.png
image/review_length_distribution.png
image/general_sentiment_scores_distribution.png
image/positive_prediction.png
image/playtime_logistic_regression.png
image/playtime_prediction.png
image/theme_mention.png
image/topic_lda.png
image/bigram_count.png
```

If your repository uses a different folder layout, update the relative image paths accordingly.

---

# Reproducibility notes

The notebook imports the following main libraries:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer
from gensim import corpora
from gensim.models import LdaModel
```

Required NLTK downloads used by the notebook:

```python
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('movie_reviews', quiet=True)
```

---

# Summary

This project turns Steam reviews for **Darkest Dungeon** into a reusable NLP pipeline:

- the raw Steam export is filtered and normalized
- text is cleaned with contraction handling and lemmatization
- `review_clean` becomes the main text field for vectorization and topic modeling
- recommendation labels and playtime fields are transformed into supervised learning targets
- theme-level scoring adds interpretable domain signals
- visual outputs summarize sentiment, topics, theme frequency, and modeling performance

This README is designed to document the **dataset structure**, the **preprocessing logic**, and the **meaning of the notebook outputs** for GitHub publication.
