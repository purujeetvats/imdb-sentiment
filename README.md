# IMDB Sentiment Classifier

Phase 6 of my ML learning roadmap: NLP (teaching a model to read text). Three models compared, from word-counting to a fine-tuned transformer.

## Problem
Read a movie review and say whether it's **positive** or **negative**.

## Data
IMDB Large Movie Review Dataset (Stanford), loaded with Hugging Face `datasets`: 25,000 training reviews and 25,000 test reviews, half positive and half negative. It downloads automatically (not committed).

## Approach
1. **TF-IDF + Logistic Regression**: counts words. The fast baseline.
2. **TextCNN from scratch**: my own tokenizer, vocabulary, embeddings and Conv1d network in PyTorch.
3. **Fine-tuned DistilBERT**: a pre-trained transformer, Hugging Face tokenizer, AdamW with warmup + linear learning-rate decay, 1 epoch.
4. **Error analysis**: tricky reviews ("not bad", "but…") run through both TF-IDF and DistilBERT.

## Results
| Model | Test accuracy |
|---|---|
| Baseline (always guess one class) | 50% |
| TF-IDF + Logistic Regression (74,849-word vocab, ~5 s on CPU) | 88.29% |
| My own TextCNN (from scratch: embeddings + Conv1d, Adam, 5–6 epochs) | ~85% (still wrong on "not bad at all") |
| Fine-tuned DistilBERT (pre-trained, 1 epoch, AdamW 3e-5 + warmup/linear schedule, ~5.5 min on GPU) | **91.33%** |

## What I learned / what failed

**Which approach are we following?** Both, in order. It's the same habit as every phase: a simple baseline first, then something stronger, and compare.
1. **TF-IDF + Logistic Regression** (counts words). Fast, and sets the score to beat (~89%).
2. **My own text neural network, built from scratch in PyTorch.** I write the tokenizer, the vocabulary and the embedding layer myself, so nothing is hidden.
3. **DistilBERT fine-tuning** (understands context). Should beat both (~92–93%).
4. ~~Deploy the winner as a Gradio app on Hugging Face Spaces.~~ Skipped: I chose to wrap up and move on to LLM APIs. The trained model is saved locally in `models/`.

**Why not build DistilBERT from scratch?** Its "knowledge of English" comes from pre-training on billions of words, which takes days on many big GPUs. Nobody does that for a single task. **Transfer learning** = take a model someone already pre-trained and fine-tune it. Step 2 teaches what's inside (tokens, embeddings); step 3 teaches how real projects reuse big models.

If DistilBERT only beats word-counting by a little, that's a real finding too: simple can be good enough.

**Loading the data (Cell 1):**

| Function | What it does | Why it helps |
|---|---|---|
| `load_dataset("stanfordnlp/imdb")` | gets the dataset from Hugging Face (cached after the first time) | one line; the full name **owner/name** is required in new versions |
| `imdb["train"]` / `imdb["test"]` | picks a split | 25k train and 25k test (plus 50k unlabeled "unsupervised" reviews I don't use) |
| `train.features["label"].names` | the label names | 0 = neg, 1 = pos |
| `train[0]["text"][:400]` | the first 400 characters of review 0 | a quick look |

- **Hugging Face** = "GitHub for AI": datasets, pre-trained models, and app hosting (Spaces).
- The text is raw and messy, with HTML leftovers like `<br />`, and every review is a different length.
**Exploring (Cell 2):**
- **Balanced:** 12,500 neg / 12,500 pos, so accuracy is fair and random guessing = 50%.
- ⚠️ **Sorted:** all negatives come first, then all positives. Always shuffle before taking a subset, or a "quick 5,000-review sample" would be 100% negative.
- **Lengths:** min 10, median 174, mean 234, max 2,470 words. DistilBERT reads at most ~512 tokens, so long reviews get cut. That's the small version of an LLM's **context window**.
- `<br />` HTML appears in 14,665 reviews.

| Function | What it does |
|---|---|
| `Counter(list)` | counts each value |
| `text.split()` | splits at spaces, giving words (the simplest tokenizer) |
| `[f(x) for x in list]` | list comprehension: applies something to every item |
| `"<br />" in text` | True if the text contains it |
| `sum(...)` of True/False | counts the Trues |

**TF-IDF (Cell 3):** turns text into numbers, with one column per vocabulary word.
- **TF** = how often the word appears in *this* review.
- **IDF** = how rare the word is across all reviews ("the" → low, "masterpiece" → high).
- Score = TF × IDF.
- Every review becomes a row of the **same length**, which solves the different-lengths problem.
- **What's lost: word order** ("not good" = "good, not").
- The output is **sparse** (mostly zeros, stored compressed).

| Function | What it does |
|---|---|
| `TfidfVectorizer()` | the text → numbers tool (tokenize, lowercase, count, weight) |
| `fit_transform(texts)` | learn the vocabulary + IDF, then convert |
| `get_feature_names_out()` | the vocabulary (column names) |
| `.toarray()` | sparse → normal table, to look at |

**Baseline (Cell 4):** `Pipeline([("tfidf", TfidfVectorizer()), ("model", LogisticRegression(max_iter=1000))])` → **88.29%** test accuracy in ~5 seconds on CPU.

| Function | What it does |
|---|---|
| `Pipeline([...])` | text → TF-IDF → model; TF-IDF is learned from train only |
| `named_steps["tfidf"]` | reaches into the pipeline by step name |
| `.vocabulary_` | word → column dictionary (74,849 words) |
| `accuracy_score(true, pred)` | fraction correct |

- Bug: a missing comma between pipeline steps → "'tuple' object is not callable; perhaps you missed a comma?" Python reads `(a, b)(c, d)` as calling a tuple.
- Tip: end every line of a multi-line list with a comma.

**Inside the baseline (Cell 5):**
- Most negative words: worst −9.4, bad −7.7, awful −6.6, waste −6.4, boring −6.2.
- Most positive words: great +7.4, excellent +6.4, best +5.1, perfect +5.0, wonderful +4.8.

It learned these by itself from 25k reviews. My own reviews:
- "masterpiece... loved" → 91% positive ✅
- "boring, badly acted" → 4% ✅
- **"I expected to hate it, but honestly it was not bad at all" → 4% positive ❌ (confidently wrong).** It only counts words: it sees "hate" and "bad", and has no idea that "not" flips "bad". That's the cost of losing word order.

| Function | What it does |
|---|---|
| `.coef_[0]` | learned weights (one per word) |
| `np.argsort(x)` | positions that would sort x (smallest first) |
| `list[::-1]` | reversed list |
| `predict_proba(texts)` | [P(neg), P(pos)] per text |
| `zip(a, b)` | walk two lists side by side in pairs |

- `np.float64(-9.4)` in the printout = numpy's way of showing its number type. `float(x)` prints plain `-9.4`.

**My own tokenizer + vocabulary (Cell 6):**
- **Tokenize:** lowercase, remove `<br />`, then `re.findall(r"[a-z0-9']+", text)` grabs runs of letters/digits/apostrophes and drops punctuation.
- **Vocabulary:** a dictionary word → ID. `<pad>`=0 (filler), `<unk>`=1 (unknown word), then the 19,998 most common words by frequency (the=2, bad=76, great=85).
- 87,171 distinct words are cut to 20,000; rare ones become `<unk>`.
- **Encode:** "Not bad at all, a masterpiece of zxqv!" → `[21, 76, 30, 29, 4, 990, 5, 1]`. **Order is kept**, unlike TF-IDF.
- 🔗 Agents: LLMs read **token IDs** exactly like this (their tokens are word pieces, with ~100k+ vocab). API "tokens" = these IDs.

| Function | What it does |
|---|---|
| `re.findall(pattern, text)` | every piece of text matching the pattern |
| `Counter().update(list)` | adds counts for every item |
| `.most_common(n)` | top-n (item, count) pairs |
| `dict.get(key, default)` | lookup with a fallback (unknown → 1) |

**Padding + batches (Cell 7):** every review is cut or filled to exactly 256 IDs (`<pad>`=0), because GPUs need rectangular blocks. 28% of reviews are longer than 256 and get cut.

`X_train = torch.tensor([pad(encode(t)) for t in train["text"]])`, read from the inside out, for each review `t`:
1. `encode(t)`: text → list of IDs, any length.
2. `pad(...)`: → exactly 256 IDs.
3. `[... for t in train["text"]]`: done for all 25,000 reviews, giving a list of 25,000 lists.
4. `torch.tensor(...)`: turns that into one `[25000, 256]` tensor that PyTorch/GPU can use.

It's the text version of `ToTensor()` in Fashion-MNIST: all the conversion is done once, up front.

`print(pad([21, 76, 30, 29])[:8], ..., len(...))` is just a **test** of `pad` on a tiny example ("not bad at all" = 4 IDs). It shows the first 8 values (`[21, 76, 30, 29, 0, 0, 0, 0]`) and checks the length is 256. It's good practice to test a new function on something small before using it on 25,000 reviews.

| Function | What it does |
|---|---|
**The 4 tensors:** X = the inputs (reviews as IDs), y = the answers (labels). train = learn from, test = final exam.

| Tensor | Holds | Shape |
|---|---|---|
| `X_train` | 25k training reviews, each as 256 word IDs | `[25000, 256]` |
| `y_train` | their answers (0 = neg, 1 = pos) | `[25000]` |
| `X_test` | 25k test reviews as 256 IDs | `[25000, 256]` |
| `y_test` | their answers | `[25000]` |

**What are we predicting?** y: is the review positive (1) or negative (0), from its text alone.
- **Training:** the model sees X_train and y_train and learns the link.
- **Testing:** it sees only X_test and predicts; I compare with y_test (which the model never sees) to get the accuracy.
- **Real use (the app):** someone types a brand-new review with **no answer**, and the model's prediction *is* the answer.

Row i of X goes with item i of y. It's the same X/y idea as Titanic, now as tensors.

| `TensorDataset(X, y)` | pairs each review with its label for the DataLoader |
| `[0] * n` | a list of n zeros |

**Embeddings (Cell 8):** `nn.Embedding(20000, 64, padding_idx=0)` is a learnable table with one row of 64 numbers per word ID: the word's "meaning coordinates". It's random at first; training moves words used alike close together (great ≈ excellent). TF-IDF can't do that: every word is its own unrelated column. `padding_idx=0` keeps `<pad>`'s row at zeros. Shape: `[1, 4]` IDs → `[1, 4, 64]`. 🔗 Agents: RAG = sentence embeddings + "find the closest ones"; every LLM starts with an embedding table.

My questions:
1. **Why `vectors[0, 0, :5]` when there's only 1 review?** The shape is `[1, 4, 64]` = [review, word, number]. Even with one review the "review" dimension still exists (layers always work on batches), so I need all three positions: review 0, word 0 ("not"), first 5 numbers.
2. **Which 20,000 words?** The 19,998 most frequent words in the 25k training reviews (from Cell 6's `most_common`) + `<pad>` + `<unk>`. The cut-off words appear only ~10 times (e.g. "rattling", "moniker", "hoop"). These 20k words cover **97.2%** of all word occurrences; the other 67k words are rare.
3. **What if no `.detach()`?** The values are the same; the print just adds `grad_fn=<SliceBackward0>` (PyTorch's note that it's tracking how this was computed, for learning). But converting to numpy fails: "Can't call numpy() on Tensor that requires grad". `.detach()` = "just the numbers, no learning history".
4. **Why `torch.tensor(...)`?** PyTorch layers only accept tensors, not Python lists: `emb([21, 76])` → "TypeError: ... must be Tensor, not list". A tensor is a fixed-shape grid of numbers that can go to the GPU and track gradients. The extra `[ ]` in `torch.tensor([encode(...)])` makes it a batch of 1: `[1, 4]`, not `[4]`.

**How does the embedding know the 20,000 words? It doesn't.** It only knows row numbers 0–19,999. The words live in my `vocab` dictionary (Cell 6):
```
"not bad"  --encode() uses vocab-->  [21, 76]  --embedding-->  row 21, row 76  (64 numbers each)
```
- `nn.Embedding(20000, 64)` just means "make 20,000 empty rows". 20000 must equal `len(vocab)`.
- The *meaning* of each row comes from training: row 76 shows up in negative reviews, so its numbers drift toward "negative".

**But I never gave it 20,000 words, so how did it work on them?** It didn't. `nn.Embedding(20000, 64)` only **creates** a blank `[20000, 64]` table of random numbers (`emb.weight`). Nothing is fetched. `emb(ids)` then just **reads the requested rows** (e.g. 21, 76, 30, 29) and ignores the other 19,996. `emb.weight[76]` is exactly the 64 numbers that came out for "bad". The words only ever exist in `vocab`/`encode()`.

It's like a coat check: `vocab` is the ticket desk (word → number), and the embedding is a rack of 20,000 hooks that only knows hook numbers.

**My own network: TextCNN (Cell 9).** I wrote it as a class because `permute` and `max` aren't layers, so `Sequential` can't do them.
- `__init__` = parts list (the layers it owns, stored as `self.x`); `forward` = the order of steps; `super().__init__()` is required setup.
- Flow: IDs `[64, 256]` → embed `[64, 256, 64]` → permute `[64, 64, 256]` → `Conv1d(64, 128, kernel_size=3)` = 128 detectors looking at **3 consecutive words** `[64, 128, 256]` → max over words ("did the pattern appear anywhere?") `[64, 128]` → dropout → `Linear(128, 2)` → `[64, 2]`.
- 1,304,962 weights; 1.28M of them are the embedding table.

| Function | What it does |
|---|---|
| `class X(nn.Module)` | my own model blueprint |
| `x.permute(0, 2, 1)` | reorders dimensions (shape only) so Conv1d gets [batch, channels, length] |
| `x.max(dim=2).values` | the max along the words dimension |

**How the data flows (visual):**
```
imdb (DatasetDict)
 ├─ "train" → 25,000 rows of {"text": "...", "label": 0/1}
 ├─ "test"  → 25,000 rows of {"text": "...", "label": 0/1}
 └─ "unsupervised" (ignored)

text → tokenize → encode (vocab) → pad → torch.tensor
X_train [25000, 256]   y_train [25000]
   ↓ TensorDataset pairs them, DataLoader cuts them into batches
x [64, 256]  +  y [64]   × 391 batches per epoch
```

**Why my own network lost to TF-IDF (~85% vs 88.3%):** it had to learn what every word means from zero, from only 25k reviews, so it overfit (train > test). TF-IDF only counts words, which works fine with little data. The fix isn't a bigger network; it's a model that already knows English → **pre-training** (DistilBERT). Data beats architecture. That's why the AI industry is built on pre-trained models.

**Reading code: the 8 patterns** that cover almost everything:
1. `name = Thing(...)`: create and store
2. `thing.action()`: call a method
3. `def f(blanks): ... return x`: a recipe
4. `a, b = x, y`: unpacking (counts must match: `correct, total = 0` fails, `= 0, 0` works)
5. `for x in things:`: a loop (indentation = what repeats)
6. `x += n`: add to what's there
7. `[f(x) for x in things]` / `{k: v for ...}`: build a list/dict in one line
8. `a.b().c()`: chaining (read the dots as "and then")

**A batch:** `x` = `[64, 256]` grid (64 reviews × 256 word IDs, padding 0s at the end of short ones); `y` = `[64]` answers. Thanks to the shuffle, each batch is a mix of 0s and 1s. 391 batches = 1 epoch.

**What's the attention mask?** A list of 1s and 0s the same length as the token IDs: **1 = real token, read it; 0 = padding, ignore it.**
```
tokens: [CLS] not  bad  at  all [SEP] [PAD] [PAD] [PAD]
ids:     101 2025 2919 2012 2035 102    0     0     0
mask:      1    1    1    1    1   1    0     0     0
```
Without it, DistilBERT would treat the padding as part of the review: a short review padded to 256 would be mostly "empty words", which would blur its meaning. My TextCNN had no mask, so it had to learn by itself that ID 0 means nothing. The name comes from "masking out" (covering up) the parts to ignore, like masking tape when painting.

**Why batch 16 for training but 64 for testing?** Memory.
- **Training** must keep every intermediate result for `backward()`, plus the gradients, plus Adam's 2 extra numbers per weight. That's several copies of a 67M-weight model, so only ~16 long reviews fit in 8 GB of GPU memory.
- **Testing** runs under `torch.no_grad()`, so nothing is kept for learning. Much less memory → bigger batches → faster.

Batch size **doesn't change test results**, only speed. In training it does affect learning (16 = more, smaller nudges per epoch), and 16–32 is the usual choice for fine-tuning.

**Does Logistic Regression give weights or probabilities? Both, at different stages:**
1. **Weights** are what it *learns* during `fit` (one per word, e.g. great +6, awful −7). They live in `.coef_`.
2. For a review: score = Σ (word's TF-IDF × word's weight) + bias.
3. **Sigmoid** squashes the score into a **probability** from 0 to 1. `predict_proba` shows it.
4. `predict` = probability > 0.5 → 1 (pos), else 0 (neg).

It's the same as Titanic: there, Sex had weight 2.559, and the output was still a survival probability.

**When did we split train/test?** We didn't. The dataset's creators (Stanford) published it already split: 25k train / 25k test (+50k unlabeled), each balanced 50/50. `imdb["train"]` and `imdb["test"]` just pick the ready-made parts. That's the same as Fashion-MNIST (`train=True/False`) and Kaggle (train.csv/test.csv). In Titanic I split myself with `train_test_split` because I needed a validation set. Ready-made splits mean everyone tests on the same reviews, so scores are comparable across papers.

**DistilBERT's tokenizer (Cell 11):** it splits words into **subword pieces**, so it never meets an unknown word: "unbelievably" → `un ##bel ##ie ##va ##bly` (`##` = "glued to the previous piece"). It adds `[CLS]` (101) at the start, `[SEP]` (102) at the end, and `[PAD]` (0) as filler. It also returns the attention mask.

**Tokenizing all 50k reviews (Cell 12):**
```
train (25,000: text, label)
  │ .map(bert_tokenize, batched=True)  → 1,000 reviews at a time → bert_tok → return!
  ▼
bert_train (+ input_ids, attention_mask, 256 each)
  │ .set_format("torch", columns=cols)
  ▼
DataLoader(batch_size=16, shuffle=True) → 1,563 batches
  ▼
b = {"input_ids": [16,256], "attention_mask": [16,256], "label": [16]}
```
| Function | What it does | How it helps |
|---|---|---|
| `def bert_tokenize(batch): return ...` | recipe that tokenizes one chunk of reviews | `map` calls it ~25 times. **Without `return`, it gives back nothing and no columns get added** (my bug) |
| `dataset.map(fn, batched=True)` | runs `fn` over the whole dataset in chunks and adds what it returns as new columns | tokenizes 25k reviews in one line (the 25000/25000 progress bar) |
| `.set_format("torch", columns=...)` | makes the dataset hand out those columns as tensors | the model only accepts tensors, and only needs those 3 columns |
| `DataLoader(...)` | groups rows into batches (shuffled for train) | each batch arrives as a **dictionary** of tensors |
| `next(iter(loader))` | takes one batch out | quick shape check before the slow training |

**Didn't we already load DistilBERT in Cell 11?** No, only its **tokenizer**. One name (`"distilbert-base-uncased"`) holds two separate parts:
| Part | Loaded with | Job | Weights? |
|---|---|---|---|
| Tokenizer (Cell 11) | `AutoTokenizer.from_pretrained` | text → token IDs (a fixed word list + splitting rules) | none, it doesn't learn |
| Model (Cell 13) | `AutoModelForSequenceClassification.from_pretrained` | token IDs → neg/pos | 67M, and it learns |

It's the same split as my own version: `vocab`/`encode()` (Cell 6) turned text into IDs, and `TextCNN` (Cell 9) was the model. They must come from the same name, so the IDs mean the same thing to both.

**Body vs head (what `AutoModelForSequenceClassification` builds):** every classifier has 2 parts. My TextCNN already did:
```
TextCNN:    review IDs ──► [embed → conv → relu → max]  ──► 128 numbers ──► [fc: Linear(128, 2)] ──► 2 scores
DistilBERT: review IDs ──► [6 transformer layers]       ──► 768 numbers ──► [Linear(768, 2)]     ──► 2 scores
                           └──── BODY: understands ────┘   "summary"       └─ HEAD: decides ─┘
```
The **body** is pre-trained (Google trained it on fill-in-the-blank over billions of words), so it already makes good summaries. The **head** is new and random, because Google never asked "neg or pos?". `num_labels=2` = how many outputs the head has (Fashion-MNIST would need 10). Training mostly teaches the head and slightly adjusts the body. "Sequence classification" = a whole text in → one class out.

**What does the LOAD REPORT mean?** The downloaded file is a list of **named weight blocks**. My model has a list of **named slots**. `from_pretrained` matches them by name:
```
DOWNLOADED FILE (Google's)            MY MODEL (needs)
body blocks (100s)        ── match ──► body slots        → filled with Google's weights ✓
vocab_transform/...       ── no slot ─► (nothing)         → UNEXPECTED: thrown away
(nothing)                 ── no file ─► pre_classifier,   → MISSING: filled with random numbers
                                        classifier
```
UNEXPECTED = the old fill-in-the-blank head (not needed). MISSING = my new neg/pos head (`pre_classifier` Linear(768,768) + `classifier` Linear(768,2)), which is random until training. Normal for transfer learning. It would only be a problem if **body** names showed up as MISSING (the body wouldn't have loaded).

**Why `[:, 1]`?** `predict_proba` / `softmax` give a grid: **one row per review, 2 columns: [chance neg, chance pos]**. `[rows, columns]` picks parts of it. `:` = "all rows", `1` = "column 1" (counting from 0, so column 0 = neg, column 1 = pos). So `[:, 1]` = "for every review, give me just the % positive." The neg column isn't needed, because neg = 100% − pos.
```
              col 0 (neg)  col 1 (pos)
review 0  [   0.09         0.91   ]
review 1  [   0.96         0.04   ]      [:, 1] → [0.91, 0.04, 0.96]
review 2  [   0.04         0.96   ]
```

**Head-to-head on tricky reviews (Cell 16)**, % positive:
| Review | TF-IDF | DistilBERT | Truth |
|---|---|---|---|
| An absolute masterpiece. I loved every minute of it. | 91 ✓ | 99 ✓ | pos |
| Two hours I will never get back. Boring and badly acted. | 4 ✓ | 1 ✓ | neg |
| I expected to hate it, but honestly it was not bad at all. | 4 ✗ | 15 ✗ | pos |
| Not good. Not good at all. | 53 ✗ | 3 ✓ | neg |
| The acting was terrible and the plot was a mess, but somehow I loved it. | 3 ✗ | 83 ✓ | pos |
| I wanted to love this movie, but it was a huge disappointment. | 12 ✓ | 3 ✓ | neg |

TF-IDF counts words, so "not good" looks half positive and "terrible…mess…but loved it" looks negative. DistilBERT reads order and context, so it got 5 of 6. It still missed "expected to hate it… not bad at all": three negative-sounding words ("hate", "not", "bad") in a short, unusual sentence, after only 1 epoch. **91% accuracy ≠ never wrong.** Always look at the failures (error analysis), not just the score.

**Why `predict_proba` for TF-IDF but `softmax` for BERT?** It's the same job (scores → percentages), done by different libraries:
```
scikit-learn:  tfidf_model.predict_proba(texts)        → [n, 2] percentages   (the conversion is done INSIDE)
PyTorch:       bert_model(...).logits → softmax(...)    → [n, 2] percentages   (the model gives raw scores; I convert them myself)
```
scikit-learn is built for convenience: `predict` = answer, `predict_proba` = percentages. PyTorch models only output raw scores (logits), because training needs them raw: the loss function does its own conversion inside, more precisely. So for percentages I call `softmax` myself. Both end as the same grid, so `[:, 1]` works on both.

- **`'NoneType' object has no attribute 'backward'` (Cell 14):** I wrote `label=y` instead of `labels=y`. Hugging Face models quietly accept unknown keyword names, so no error came at that line. Without `labels`, the model has no answer key, so `out.loss` = `None`, and the crash only shows up one line later at `.backward()`. **One letter, and the error appears somewhere else.**
- **`name 'argmax' is not defined` (Cell 15):** I wrote `logits(argmax(dim=1))` instead of `logits.argmax(dim=1)`. **Dot = "ask this thing to do its action"** (pattern 2). **Brackets around something = "hand it to a function".** Python works from the inside out, so it looked for a standalone function called `argmax`, and there isn't one. `argmax` only exists *on* tensors: `logits.argmax(...)`.
- **"Unauthenticated requests to the HF Hub" warning:** harmless. Without a free account token, downloads just have lower rate limits. I'll make a token at the deploy step.

## Run it
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
```
Uses an NVIDIA GPU if available (CUDA 13.0 build of PyTorch).
