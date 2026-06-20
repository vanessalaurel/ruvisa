# Review Analyzer — Expanded Materials & Method

Expanded version of the review analyzer subsection, with dataset statistics, complete keyword and regex inventories, the sentiment classification mechanism, worked examples, skin type suitability detection, and suggested figures.

---

## 1. Dataset

### Review corpus

The review corpus consists of **5,965 user reviews** scraped from Sephora, covering **872 products** in the catalogue. Each review record contains:

| Field | Description |
| --- | --- |
| `review_text` | The full body text of the review |
| `headline` | An optional short summary written by the reviewer |
| `rating` | Star rating on a 1–5 scale |
| `product_url` | URL linking the review to its product in the catalogue |

Both `review_text` and `headline` are concatenated into a single text string for analysis, maximizing the amount of signal available from each review.

### Review distribution across products

The 5,965 reviews are distributed across 872 products, yielding a mean of approximately **6.8 reviews per product**. The distribution is right-skewed: a small number of popular products accumulate many reviews, while the majority of products have only a handful. Products with very few reviews naturally yield less reliable aggregate scores; when no reviews mention a particular concern, the system falls back to ingredient evidence alone (see Product Evidence Vector).

> **Suggested Figure A — Reviews per product distribution.** A histogram with x-axis = number of reviews per product and y-axis = number of products. Annotate the mean and median. This shows the reader that the review signal is sparse for most products, motivating the fusion with ingredient evidence.

---

## 2. Approach

Rather than training a neural model on the relatively small and noisy review corpus, we adopt a **keyword-based concern detection with context-aware sentiment classification** approach. This avoids the need for labeled review data while maintaining interpretability. The pipeline has three stages.

---

## 3. Stage 1 — Concern Detection

For each review, the system identifies which of the 7 skin concerns are mentioned using two complementary methods applied in sequence.

### 3.1 Keyword matching (primary)

Each concern has a curated keyword list. If any keyword appears as a substring in the lowercased review text, that concern is flagged as mentioned.

**Table 1: Complete concern keyword inventory**

| Concern | Keywords | Count |
| --- | --- | ---: |
| acne | acne, pimple, pimples, breakout, breakouts, break out, breaking out, broke out, blemish, blemishes, zit, zits, cystic acne | 13 |
| comedonal_acne | blackhead, blackheads, whitehead, whiteheads, comedone, comedones, comedonal, clogged pore, clogged pores | 9 |
| pigmentation | dark spot, dark spots, hyperpigmentation, pigmentation, discoloration, discolouration, melasma, sun spot, sun spots, uneven tone, uneven skin tone, dark mark, dark marks | 13 |
| acne_scars_texture | acne scar, acne scars, scarring, uneven texture, skin texture, rough skin, roughness, bumpy skin, ice pick | 9 |
| pores | my pore, my pores, large pore, enlarged pore, visible pore, open pore, minimize pore, minimise pore, refine pore, tighten pore, shrink pore, clogged pore, clogged pores | 13 |
| redness | redness, rosacea, my skin red, face red, inflamed, inflammation | 6 |
| wrinkles | wrinkle, wrinkles, fine line, fine lines, anti-aging, anti aging, antiaging, crow feet, crow's feet, laugh line, laugh lines, sagging | 12 |

Keywords are curated to capture the colloquial language reviewers use (e.g., "zit" and "cystic acne" alongside the clinical term "acne"). Multi-word phrases (e.g., "clogged pore", "dark spot") are included to reduce false positives from single-word matches.

### 3.2 Context-aware regex patterns (supplementary)

For concerns where simple keywords may miss contextual mentions, supplementary regex patterns are applied after keyword matching. These patterns capture discussion of a concern without using the concern's canonical keywords — for example, a reviewer saying *"it brightened my skin"* is discussing pigmentation even though the word "pigmentation" never appears.

**Table 2: Supplementary concern regex patterns**

| Concern | Regex pattern | What it captures |
| --- | --- | --- |
| pigmentation | `brighten\w*\s+(?:my\s+)?(?:skin\|face\|complexion\|dark)` | "brightened my skin", "brightens dark complexion" |
| acne_scars_texture | `(?:my\|skin)\s+(?:texture\|scars?)` | "my texture", "skin scars" |
| acne_scars_texture | `(?:smooth\|soften)\w*\s+(?:my\s+)?(?:skin\|face)` | "smoothed my skin", "softens face" |
| pores | `(?:my\|visible\|large\|big\|huge\|open\|clogged\|enlarged)\s+pores?` | "my large pores", "visible pores" |
| pores | `pores?\s+(?:look\|appear\|are\|seem\|feel)\s+(?:smaller\|bigger\|larger\|cleaner\|clearer\|tighter)` | "pores look smaller", "pores feel tighter" |
| redness | `(?:calm\|soothe\|reduc)\w*\s+(?:my\s+)?(?:redness\|irritation\|inflammation)` | "calmed my redness", "reduces irritation" |
| redness | `(?:skin\|face)\s+(?:got\|became\|turn(?:ed)?\|is\|was)\s+red` | "my skin turned red", "face got red" |
| redness | `(?:irritat\|sting\|burn)\w+\s+(?:my\s+)?(?:skin\|face)` | "irritated my skin", "stinging my face" |
| wrinkles | `(?:firm\|plump\|tighten)\w*\s+(?:my\s+)?(?:skin\|face)` | "firms my skin", "plumped my face" |
| wrinkles | `(?:aging\|ageing)\s+(?:skin\|concern\|sign)` | "aging skin", "signs of ageing" |

There are **10 supplementary regex patterns** across **5 concerns**. Acne and comedonal_acne do not require supplementary patterns because their keyword lists are already sufficiently distinctive — reviewers almost always use explicit terms like "breakout" or "blackhead" when discussing these concerns.

The output of Stage 1 is a **set of detected concerns** for the review (a subset of the 7 possible concerns). Concerns not mentioned receive a label of 0 and are not processed further.

---

## 4. Stage 2 — Sentiment Classification

For each detected concern, context-aware regex templates classify the sentiment as **positive** (the product improved the concern) or **negative** (the product worsened the concern).

### 4.1 The word gap mechanism

Sentiment words and concern keywords rarely appear adjacent in natural text. To handle intervening words, all templates use a **word gap** fragment:

$$W = \texttt{(?:\\s+\\S+)\{0,5\}\\s+}}$$

This regex matches **0 to 5 intervening words** between two terms. For example, the template `help(s|ed)? W {kw}` matches:

- *"helped my **acne**"* (1 intervening word)
- *"helped clear up my **acne**"* (3 intervening words)
- *"helped with my stubborn cystic **acne**"* (4 intervening words)

but would **not** match *"helped me feel more confident about my skin and cleared my acne"* (too many words between "helped" and "acne"), avoiding false positives from long-range co-occurrences.

### 4.2 Positive sentiment templates (18 patterns)

Each template contains a `{kw}` placeholder that is instantiated with every keyword from the detected concern's keyword list. The `...` below represents the word gap W.

**Table 3: Positive sentiment templates**

| # | Template | Example match |
| ---: | --- | --- |
| 1 | `help(s\|ed)? ... {kw}` | "helped my **acne**" |
| 2 | `help(s\|ed)?\s+(?:with\s+)?(?:my\s+)?{kw}` | "helps with my **breakouts**" |
| 3 | `reduc(e[ds]?\|ing) ... {kw}` | "reduced my **redness**" |
| 4 | `clear(s\|ed\|ing)?\s+(?:up\s+)? ... {kw}` | "cleared up my **acne**" |
| 5 | `(?:got\|get\|getting)\s+rid\s+of ... {kw}` | "got rid of **dark spots**" |
| 6 | `improv(e[ds]?\|ing) ... {kw}` | "improved my **wrinkles**" |
| 7 | `diminish(es\|ed\|ing)? ... {kw}` | "diminished my **fine lines**" |
| 8 | `(?:less\|fewer\|no\s+more\|minimize[ds]?)\s+{kw}` | "less **redness**", "no more **breakouts**" |
| 9 | `{kw} ... (?:went\s+away\|disappeared\|cleared\|improved\|gone\|reduced\|faded)` | "**acne** cleared", "**dark spots** faded" |
| 10 | `(?:fad(e[ds]?\|ing)\|lighten(s\|ed\|ing)?) ... {kw}` | "fading **dark spots**" |
| 11 | `(?:great\|good\|amazing\|excellent\|perfect\|fantastic\|love\|wonderful)\s+for\s+{kw}` | "great for **acne**" |
| 12 | `no\s+(?:more\s+)?{kw}` | "no more **pimples**" |
| 13 | `(?:prevent\|prevents\|prevented\|preventing) ... {kw}` | "prevented **breakouts**" |
| 14 | `{kw}\s+(?:are\|is\|was\|were)\s+(?:much\s+)?(?:better\|less\|smaller\|fewer\|gone)` | "**pores** are smaller" |
| 15 | `(?:healed?\|healing) ... {kw}` | "healed my **blemishes**" |
| 16 | `(?:fight\|fights\|combat\|combats) ... {kw}` | "fights **acne**" |
| 17 | `(?:tighten\|minimiz\|shrink\|refin)\w* ... {kw}` | "minimized my **pores**" |
| 18 | `(?:smooth\|soften)\w* ... {kw}` | "smoothed my **rough skin**" |

### 4.3 Negative sentiment templates (7 patterns)

**Table 4: Negative sentiment templates**

| # | Template | Example match |
| ---: | --- | --- |
| 1 | `(?:caus\|gave\|give\|trigger)\w* ... {kw}` | "caused **breakouts**" |
| 2 | `(?:made\|make\|making) ... {kw}\s+worse` | "made my **acne** worse" |
| 3 | `(?:more\|increased?\|worsen)\s+{kw}` | "more **redness**" |
| 4 | `{kw}\s+(?:got\|became\|become)\s+worse` | "**acne** got worse" |
| 5 | `{kw}\s+(?:increased\|appeared\|flared\|worsened)` | "**redness** flared" |
| 6 | `(?:didn.t\|did\s+not\|doesn.t\|does\s+not\|won.t)\s+help ... {kw}` | "didn't help my **acne**" |
| 7 | `no\s+(?:effect\|improvement\|change\|difference) ... {kw}` | "no improvement in **dark spots**" |

### 4.4 Scoring

For each detected concern, the system iterates over all of that concern's keywords. For each keyword present in the text, it tests every positive and negative template (substituting the keyword for `{kw}`), counting the number of matches:

$$\text{pos\_score} = \sum_{\text{kw} \in K_c} \sum_{t \in T^+} \mathbb{1}[\text{match}(t, \text{kw}, \text{text})]$$

$$\text{neg\_score} = \sum_{\text{kw} \in K_c} \sum_{t \in T^-} \mathbb{1}[\text{match}(t, \text{kw}, \text{text})]$$

where $K_c$ is the keyword set for concern $c$, and $T^+$, $T^-$ are the positive and negative template sets. The label is then:

$$\text{label}_c = \begin{cases} +1 & \text{if pos\_score} > \text{neg\_score} \\ -1 & \text{if neg\_score} > \text{pos\_score} \\ 0 & \text{(tie — passed to Stage 3)} \end{cases}$$

### 4.5 Special override: "broke out"

The phrase "broke out" / "breaking out" / "break out" is a strong colloquial signal for a negative acne experience. The system applies a post-hoc override:

- If the text matches `(?:broke|break|breaking)\s+(?:me\s+)?out`, the acne label is forced to **−1**.
- **Exception:** if the match is preceded (within 3 words) by a negation word — `stop`, `no more`, `prevent`, `didn't`, `doesn't`, `does not`, `don't`, `do not`, `without`, `never`, `won't`, `will not`, `hasn't`, `not` — the override is suppressed (e.g., *"stopped breaking out"* remains positive).

The negation check regex:

```
(?:stop|no\s+more|prevent|didn.t|doesn.t|does\s+not|don.t|do\s+not
|without|never|won.t|will\s+not|hasn.t|not)\s+(?:\S+\s+){0,3}
(?:break|broke|breaking)\s+(?:me\s+)?out
```

> **Suggested Figure B — Review analyzer pipeline flowchart.** A top-to-bottom flowchart showing: Input (raw review) → Stage 1 (concern detection via keywords + regex) → Stage 2 (sentiment classification via positive/negative templates) → Stage 3 (star-rating tie-breaking) → Output (per-review label vector). Each stage box should list the key elements (keyword counts, template counts, tie-break rules). See generated figure `review_pipeline_flowchart.png`.

---

## 5. Stage 3 — Tie-Breaking

When a concern is mentioned but the positive and negative pattern counts are equal (pos_score = neg_score), the star rating resolves the ambiguity:

| Star rating | Condition | Assigned label |
| :---: | --- | :---: |
| ★★★★ or ★★★★★ | rating ≥ 4 | +1 (positive) |
| ★ or ★★ | rating ≤ 2 | −1 (negative) |
| ★★★ | rating = 3 | 0 (neutral) |

If a review's default rating is missing, it defaults to **3** (neutral).

A second tie-breaking pass also applies: after all concern labels and the "broke out" override are computed, any concern that was **detected** (mentioned in Stage 1) but still has a label of **0** gets a final rating-based assignment using the same rules above. This ensures that a review mentioning a concern is rarely left without a directional signal.

---

## 6. Worked Examples

### 6.1 Example 1 — Positive review (straightforward)

**Review input:**

| Field | Value |
| --- | --- |
| review_text | *"This serum really helped clear up my acne and reduced my dark spots significantly. My pores look smaller too!"* |
| rating | 5 |
| product | Example Serum X |

**Stage 1 — Concern Detection:**

| Match type | Text matched | Detected concern |
| --- | --- | --- |
| Keyword | "acne" | acne |
| Keyword | "dark spots" | pigmentation |
| Regex | "pores look smaller" → `pores?\s+(?:look\|appear)\s+(?:smaller\|...)` | pores |

Detected concerns: **{acne, pigmentation, pores}**

**Stage 2 — Sentiment Classification:**

| Concern | Matched text | Template matched | pos | neg | Label |
| --- | --- | --- | ---: | ---: | :---: |
| acne | "helped clear up my **acne**" | `help(s\|ed)? ... {kw}` (positive #1) | 1 | 0 | **+1** |
| pigmentation | "reduced my **dark spots**" | `reduc(e[ds]?\|ing) ... {kw}` (positive #3) | 1 | 0 | **+1** |
| pores | "**pores** look **smaller**" | `{kw} ... (?:better\|less\|smaller\|...)` (positive #14) | 1 | 0 | **+1** |

**Stage 3 — Tie-Breaking:** Not needed; all concerns had pos > neg.

**Output label vector:**

| acne | comedonal_acne | pigmentation | acne_scars_texture | pores | redness | wrinkles |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| +1 | 0 | +1 | 0 | +1 | 0 | 0 |

### 6.2 Example 2 — Mixed review with "broke out" override and tie-breaking

**Review input:**

| Field | Value |
| --- | --- |
| review_text | *"I used this for my acne but it broke me out even more. At least my dark spots faded a bit."* |
| rating | 2 |

**Stage 1 — Concern Detection:**

| Match type | Text matched | Detected concern |
| --- | --- | --- |
| Keyword | "acne" | acne |
| Keyword | "broke out" | acne (already detected) |
| Keyword | "dark spots" | pigmentation |

Detected concerns: **{acne, pigmentation}**

**Stage 2 — Sentiment Classification:**

| Concern | Analysis | pos | neg | Label |
| --- | --- | ---: | ---: | :---: |
| acne | No positive templates match. No standard negative templates match either. | 0 | 0 | 0 (tie) |
| pigmentation | "**dark spots** faded" matches `{kw} ... (faded)` (positive #9) | 1 | 0 | **+1** |

**"Broke out" override:** "broke me out" matches the override pattern. Negation check: no negation word precedes it. → acne label is **overridden to −1**.

**Stage 3 — Tie-Breaking:** Acne already resolved by override. No remaining ties.

**Output label vector:**

| acne | comedonal_acne | pigmentation | acne_scars_texture | pores | redness | wrinkles |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| −1 | 0 | +1 | 0 | 0 | 0 | 0 |

### 6.3 Example 3 — Tie-breaking in action

**Review input:**

| Field | Value |
| --- | --- |
| review_text | *"I bought this for my wrinkles. It's okay I guess."* |
| rating | 4 |

**Stage 1:** Keyword "wrinkles" → detected concern: **{wrinkles}**

**Stage 2:** No positive or negative templates match (the reviewer mentions wrinkles but expresses no clear improvement or worsening). pos=0, neg=0 → **tie**.

**Stage 3:** Rating = 4 (≥ 4) → label = **+1** (positive).

**Output:** wrinkles = +1, all others = 0.

> **Suggested Figure C — Worked example visual.** A vertical diagram showing a single review flowing through all three stages, with highlighted keyword matches, regex match arcs showing the word gap, and the resulting label vector. See generated figure `review_worked_example.png`.

---

## 7. Skin Type Suitability Detection

Skin type suitability is detected **in parallel** with concern sentiment using an analogous three-step approach. The system identifies **5 skin types**: dry, oily, sensitive, normal, and combination.

### 7.1 Step 1 — Claim detection

Regex patterns identify whether the reviewer declares their skin type. Each skin type has **4–6 claim patterns** covering common self-identification phrasings.

**Table 5: Skin type claim patterns (representative examples)**

| Skin type | Example patterns | Matches |
| --- | --- | --- |
| dry | `(?:i\s+have\|my)\s+(?:\w+\s+){0,2}dry\s+skin` | "I have very dry skin", "my dry skin" |
| dry | `my\s+skin\s+(?:is\|was\|tends?\s+to\s+be)\s+(?:very\s+)?dry` | "my skin tends to be dry" |
| oily | `(?:i\s+have\|my)\s+(?:\w+\s+){0,2}oily\s+skin` | "I have oily skin" |
| oily | `oily[\s/-]+(?:to[\s/-]+)?(?:combo\|combination)\s+skin` | "oily to combination skin" |
| sensitive | `(?:as\s+)?(?:a\s+)?(?:someone\|person)\s+with\s+sensitive\s+skin` | "as someone with sensitive skin" |
| sensitive | `my\s+sensitive\s+skin` | "my sensitive skin" |
| combination | `(?:i\s+have\|my)\s+(?:\w+\s+){0,2}(?:combination\|combo)\s+skin` | "I have combo skin" |
| normal | `normal[\s/-]+(?:to[\s/-]+)?(?:dry\|oily)\s+skin` | "normal to dry skin" |

### 7.2 Step 2 — Suitability scoring

For each claimed skin type, positive and negative templates (with `{st}` substituted for the skin type name) classify whether the product was suitable:

**Positive suitability patterns (5 templates):**

| Template | Example match |
| --- | --- |
| `(?:great\|good\|perfect\|amazing\|love\|wonderful\|ideal\|best)\s+(?:for\|on)\s+(?:my\s+)?{st}\s+skin` | "perfect for my oily skin" |
| `(?:works?\|worked)\s+(?:really\s+)?(?:well\|great\|perfectly)\s+(?:for\|on)\s+(?:my\s+)?{st}\s+skin` | "works great on my dry skin" |
| `(?:suitable\|recommend(?:ed)?)\s+for\s+{st}\s+skin` | "recommended for sensitive skin" |
| `{st}\s+skin\s+(?:loves?\|approved)` | "oily skin approved" |
| `(?:my\s+)?{st}\s+skin\s+(?:looks?\|feels?)\s+(?:amazing\|great\|better\|hydrated\|...)` | "my dry skin feels hydrated" |

**Negative suitability patterns (6 templates):**

| Template | Example match |
| --- | --- |
| `(?:not\s+(?:good\|great\|suitable\|ideal)\|bad\|terrible)\s+for\s+{st}\s+skin` | "not good for oily skin" |
| `(?:too\s+(?:dry\|oily\|heavy\|greasy\|rich\|light))\s+for\s+(?:my\s+)?{st}\s+skin` | "too heavy for my oily skin" |
| `(?:doesn.t\|does\s+not\|didn.t\|did\s+not)\s+work\s+(?:for\|on)\s+(?:my\s+)?{st}\s+skin` | "doesn't work on my sensitive skin" |
| `(?:my\s+)?{st}\s+skin\s+(?:didn.t\|doesn.t\|does\s+not)\s+(?:like\|tolerate\|agree)` | "my sensitive skin didn't tolerate it" |
| `(?:not\s+suitable\|not\s+recommend(?:ed)?)\s+for\s+{st}\s+skin` | "not recommended for dry skin" |
| `(?:harsh\|irritating\|drying\|stripping)\s+(?:for\|on)\s+(?:my\s+)?{st}\s+skin` | "too drying for my dry skin" |

### 7.3 Step 3 — Tie-breaking

The same star-rating fallback applies: when positive and negative suitability pattern counts are equal, rating ≥ 4 → +1, rating ≤ 2 → −1, rating = 3 → 0.

---

## 8. Aggregation

Individual review labels are aggregated per product. For each concern *c*, the product-level effectiveness score is:

$$\text{eff}_c = \frac{n_c^+ - n_c^-}{n_c^+ + n_c^-} \in [-1, 1]$$

where $n_c^+$ and $n_c^-$ are the counts of positive and negative reviews mentioning concern *c*. Products with no mentions for a concern receive a **null** score (excluded from the fusion step). The same formula applies for skin type suitability:

$$\text{suit}_{st} = \frac{n_{st}^+ - n_{st}^-}{n_{st}^+ + n_{st}^-} \in [-1, 1]$$

### Review mention statistics

**Table 6: Review mention statistics (5,965 reviews)**

| Concern | Positive | Negative | Total Mentions | Positive % |
| --- | ---: | ---: | ---: | ---: |
| acne | 255 | 141 | 396 | 64% |
| wrinkles | 127 | 9 | 136 | 93% |
| redness | 88 | 46 | 134 | 66% |
| pigmentation | 101 | 6 | 107 | 94% |
| acne_scars_texture | 101 | 6 | 107 | 94% |
| pores | 59 | 9 | 68 | 87% |
| comedonal_acne | 29 | 8 | 37 | 78% |

The positive-to-negative ratio varies substantially across concerns. Acne has the most balanced distribution (64% positive), while pigmentation and acne_scars_texture are heavily skewed toward positive mentions (94%), reflecting the generally favorable outcomes reported by users for brightening and texture-improving products.

> **Suggested Figure D — Concern mention distribution.** A horizontal stacked bar chart showing positive (green) and negative (red) mention counts per concern, with the positive percentage annotated on each bar. See generated figure `concern_mention_distribution.png`.

---

## 9. Product Evidence Vector

The ingredient evidence score and review effectiveness are fused into a single 7-dimensional product evidence vector:

$$\text{score}_c^{\text{product}} = 0.5 \cdot \text{sc}_c^{\text{ing}} + 0.5 \cdot \text{sc}_c^{\text{rev}}$$

Where:

- $\text{sc}_c^{\text{ing}} \in [0, 1]$ is the INCI position-weighted ingredient evidence score
- $\text{sc}_c^{\text{rev}} = \frac{\text{eff}_c + 1}{2} \in [0, 1]$ is the review effectiveness remapped from $[-1, 1]$ to $[0, 1]$

The equal weighting (0.5 / 0.5) treats ingredient composition and real-world user outcomes as complementary evidence sources. When no reviews mention a particular concern, the product vector falls back to ingredient evidence alone, ensuring coverage even for products with sparse reviews.

---

## 10. Suggested Figures

### Figure A — Reviews per product distribution

A histogram showing:
- **x-axis:** Number of reviews per product
- **y-axis:** Number of products (frequency)
- Annotated vertical lines for mean and median
- The right-skewed shape shows most products have few reviews

**Caption:** Distribution of reviews across the 872 products in the catalogue. The right-skewed distribution motivates the fusion with ingredient evidence for products with sparse review coverage.

### Figure B — Review analyzer pipeline flowchart

A top-to-bottom flowchart with:
- **Input box:** Raw review (review_text + headline + star rating)
- **Stage 1 box:** Concern Detection — keyword matching (75 keywords across 7 concerns) + supplementary regex (10 patterns across 5 concerns)
- **Stage 2 box:** Sentiment Classification — 18 positive templates and 7 negative templates with word gap W, plus the "broke out" override
- **Stage 3 box:** Tie-Breaking — star rating fallback (≥4 → +1, ≤2 → −1, 3 → 0)
- **Output box:** Per-review label vector (7-dimensional, values in {+1, −1, 0})

**Caption:** Three-stage review analysis pipeline. Each review is processed through concern detection (keyword + regex matching), context-aware sentiment classification (template-based regex scoring), and star-rating tie-breaking to produce a 7-dimensional label vector.

### Figure C — Worked example visual

A vertical diagram showing a single review flowing through all three stages:
- The review text with **highlighted keywords** and **regex match arcs** showing the word gap between sentiment words and concern keywords
- Intermediate results at each stage (detected concerns, pattern match counts)
- The final output label vector

**Caption:** Worked example of a review processed through the three-stage pipeline. Highlighted spans show keyword matches (Stage 1) and sentiment template matches (Stage 2), with arcs indicating the word gap between matched terms.

### Figure D — Concern mention distribution (bar chart)

A horizontal stacked bar chart showing:
- Each bar = one concern
- **Green** segment = positive mentions, **red** segment = negative mentions
- Positive percentage annotated on each bar
- Total count at bar end

**Caption:** Distribution of positive and negative concern mentions across 5,965 reviews. Acne shows the most balanced distribution (64% positive), while pigmentation and acne_scars_texture are heavily skewed positive (94%), reflecting favorable user outcomes for brightening and texture products.
