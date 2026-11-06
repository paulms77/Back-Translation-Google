# Back-Translation (Google Translate)

Korean text augmentation by round-tripping a sentence through another language
and back — `ko → en → ko` — using the `googletrans` library.

Built for augmenting the training set of the
[Hansol Deco Season 2 wallpaper-defect QA competition](https://github.com/paulms77/Wallpaper-defects-knowledge-chat-service),
where the labelled Korean QA pairs were too few to fine-tune on directly.

See also: [Back-Translation-Papago](https://github.com/paulms77/Back-Translation-Papago)
— the same idea via Naver Papago, which handles Korean noticeably better.

---

## How it works

A sentence translated out of Korean and back returns with the same meaning but
different surface form — different word choice, different phrasing. For a model
learning from a small corpus, that is a genuinely new training example rather
than a duplicate.

```python
def back_translate(sentence, lang, PROB=1):
    translator = Translator()
    org_lang = translator.detect(sentence).lang      # 'ko'

    if lang is not org_lang:
        translated      = translator.translate(sentence, dest=lang).text
        translated_back = translator.translate(translated, dest=org_lang).text
    else:
        translated_back = sentence

    return translated_back
```

The source language is **detected rather than assumed**, so a row that is
already English is passed through untouched instead of being mangled.

### Pivot language choice

The notebook keeps a wider pivot list available:

```python
languages = ['ko', 'en', 'fr', 'th', 'tr', 'ur', 'ru', 'bg', 'de',
             'ar', 'zh-cn', 'hi', 'sw', 'vi', 'es', 'el', 'ja']
```

Distant pivots (Urdu, Swahili) produce more diverse paraphrases but drift further
from the original meaning. English is the default — the safest trade-off for a
domain-specific technical corpus where losing the meaning loses the label.

### Parallelism

Translation is network-bound, so rows are dispatched concurrently with
**Dask bag** rather than looped:

```python
translate_bag = bag.from_sequence(dataset[translate_column].tolist()) \
                   .map(lambda x: back_translate(x, lang=lang))

with diagnostics.ProgressBar():
    bag_completed = translate_bag.compute()
```

## Setup

```bash
git clone https://github.com/paulms77/Back-Translation-Google.git
cd Back-Translation-Google

pip install googletrans==4.0.0-rc1 pandas dask
```

> Pin the version. The stable `googletrans` release on PyPI has long been broken
> against Google's current endpoints; `4.0.0-rc1` is the one that works.

## Usage

Open [`Back_Translation_google.ipynb`](Back_Translation_google.ipynb), point it
at your CSV, and name the column to augment:

```python
data_location = '/content/drive/MyDrive/llm'
train = pd.read_csv(Path(data_location) / 'train.csv')

train_aug = back_translate_parallel1(dataset=train,
                                     translate_column='답변_1',
                                     lang='en')
```

The augmented text lands in a new `{column}_translate` column, leaving the
original intact so the two can be compared before committing to the augmented
set.

## Caveats

**It is an unofficial endpoint.** `googletrans` scrapes the web translation
service rather than using the paid API, so it rate-limits, breaks when Google
changes the frontend, and is not suitable for production. For anything that has
to keep working, use the official Cloud Translation API.

**Back-translation is lossy on technical terms.** Domain vocabulary — here,
construction and interior-finishing terms — frequently comes back as a generic
synonym or an English loanword. Augmented rows are worth eyeballing before
training on them.
