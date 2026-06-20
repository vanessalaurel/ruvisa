2.3 Product Recommender
The product recommender system is used to transform raw ingredients and people in worldwide real reviews for each product into structured evidence that can be matched to each user's skin concern profile [1], [4]. The product recommender system consists of 2 parallel analysis pipelines, the first one is the Ingredient analyzer and the second one is the review analyzer, whose outputs are then combined into a single product vector. This design follows the general logic of hybrid recommendation systems, where multiple complementary signals are fused to improve coverage, interpretability, and robustness to sparsity [4].

2.3.1 Ingredient analyzer
The product catalogue is scraped from Sephora Hong Kong (sephora.com.hk) [34], including product metadata (title, brand, price, category), INCI ingredient lists, star ratings, and user reviews. The final catalogue contains 1,009 products with valid ingredient lists.



2.3.1.1 INCI
The International Nomenclature of Cosmetic Ingredients (INCI) is a standardized naming system for cosmetic ingredients adopted internationally under regulations including the EU Cosmetics Regulation (EC 1223/2009) and the U.S. cosmetic labeling framework under the Fair Packaging and Labeling Act and 21 CFR Part 701 [24], [25]. Under these regulations, manufacturers are required to list ingredients using standardized or common ingredient names on product packaging (e.g., "Tocopherol" rather than "Vitamin E", "Ascorbic Acid" rather than "Vitamin C") [24], [25].
2.3.1.2 Why INCI matters for this system
INCI lists carry two properties that make them a reliable signal for ingredient-level analysis:
Standardized nomenclature. Because all manufacturers use the same naming convention, an ingredient like "Niacinamide" appearing in product A and product B refers to the same chemical compound, enabling exact matching across the entire catalogue without synonym resolution [24]-[26].
Concentration ordering. Under EU and most international regulations, ingredients present at concentrations above 1% must be listed in descending order of concentration. Ingredients at or below 1% may appear in any order after the above-1% group. This ordering encodes approximate concentration information directly in the list structure — the earlier an ingredient appears, the more of it the product contains. The system exploits this via position weighting [24], [25].
2.3.1.3 Scraping methodology
The scraper requests 7 primary function category pages from INCIDecoder via paginated HTML parsing [26].






Table 4: 7 primary function category pages
INCIDecoder function page
Description
anti-acne
Ingredients with acne-fighting properties
skin-brightening
Ingredients that reduce hyperpigmentation
soothing
Ingredients that reduce irritation and redness
exfoliant
Chemical and physical exfoliants
cell-communicating-ingredient
Ingredients that promote cell turnover (anti-aging)
astringent
Ingredients that tighten and minimize pores
antioxidant
Ingredients that protect against oxidative damage









Three supplementary pages (surfactant-cleansing, abrasive-scrub, moisturizer-humectant) are also scraped to expand coverage. For each page, every ingredient listed is extracted with its standardized name, yielding two output files:
ingredient_evidence.json — raw mapping: {ingredient_name: [functions]}
concern_lookup.json — mapped to the 7-concern taxonomy: {ingredient_name: [concerns]}
The final lookup contains 3,118 unique ingredient entries after merging across all function pages.Each INCIDecoder function is mapped to one or more of the 7 skin concerns in the system's taxonomy
Table 5 : 7 skin concern mapping with INCIDecoder
INCIDecoder function
Mapped concern(s)
anti-acne
acne, comedonal_acne
skin-brightening
pigmentation
soothing
redness
exfoliant
pores, acne_scars_texture, comedonal_acne
cell-communicating-ingredient
wrinkles
astringent
pores
antioxidant
pigmentation

This mapping reflects dermatological consensus: for example, exfoliants address pore congestion, textural scarring, and comedonal (non-inflammatory) acne simultaneously, so the exfoliant function maps to three concerns [1].
This produces a concern lookup table of 1,362 ingredients, each associated with one or more concerns. The per-concern coverage is:



Figure 7. Evidence Ingredients per Skin Concern




The disparity (especially acne_scars_texture with only 203 ingredients) reflects the narrower pharmacological landscape for scar-treatment actives compared to broadly functional categories like pore-targeting agents.


2.3.1.4 Ingredient Matching
For each of the 1,009 products in our catalogue, we match its INCI (ingredient) list against the concern lookup using a three-tier strategy:
Exact match after normalization (lowercasing, whitespace collapsing)
Substring forward: evidence name found within the product ingredient string (minimum 5 characters)
Substring reverse: product ingredient string found within an evidence name
Longer evidence names are matched first to prefer specific matches over generic ones.

2.3.1.4.1 INCI Position Weighting
Per INCI regulation (EU Cosmetics Regulation EC 1223/2009), ingredients present at concentrations above 1% must be listed in descending order of concentration [24]. We exploit this by applying an exponential decay weight based on list position:
        					w(i)=max(0.1,e-2.3.iN-1)


where i is the 0-indexed position and N is the total number of ingredients. This assigns:
Position 0 (first ingredient, highest concentration): w≈1.0
Mid-list: w≈0.32
Last position: w=0.1 (floor, never zero)
The decay constant −2.3 is chosen so the weight at the last position equals e-2.3≈0.1. This weighting function is defined in this work rather than copied from a prior paper. It was chosen to preserve the regulatory intuition that earlier ingredients should contribute substantially more than later ones, while ensuring that late-list ingredients still retain a non-zero contribution. For each concern c, the product's ingredient evidence score is the maximum position weight among all ingredients matched to that concern:


Figure 8. Formula of Product Evidence Score Ingredient Matching


where Mc is the set of matched ingredients for concern c. This max-based aggregation is also defined in this work. Using the maximum (rather than sum or mean) reflects the pharmacological principle that a product's effectiveness for a concern is primarily driven by its most concentrated active ingredient [1], [7].



Figure 9. INCI Position Weight Decay




Ingredients listed earlier receive higher weights, reflecting the regulatory requirement that above-1% ingredients are listed in descending concentration order. The floor at 0.1 ensures trailing ingredients still contribute.


Table 6: INCI Position-Weighted Evidence Score Statistics (across 1,009 products)
Concern
Products with Evidence
Mean Score
Min
Max
pigmentation
815
0.595
0.100
1.000
redness
713
0.619
0.100
1.000
pores
647
0.598
0.100
1.000
comedonal_acne
460
0.588
0.100
1.000
wrinkles
405
0.535
0.100
1.000
acne_scars_texture
312
0.506
0.100
1.000
acne
308
0.586
0.100
1.000



2.3.1.4.2 Three-tier ingredient matching
For each product, every ingredient in its INCI list is matched against the concern lookup table using a three-tier strategy, applied in order of priority:
Tier 1: Exact match after normalization (lowercasing, whitespace collapsing)
        e.g., "Tocopherol" == "tocopherol" 
Tier 2: Forward substring — evidence name found within the product ingredient
        (minimum 5 characters to avoid spurious partial matches)
        e.g., "Ascorbic Acid" found in "Ascorbyl Tetraisopalmitate / Ascorbic Acid Derivative" 
Tier 3: Reverse substring — product ingredient found within an evidence name
        e.g., product lists "Zinc PCA", evidence has "Zinc PCA (Zinc L-Pyrrolidone Carboxylate)" 
Longer evidence names are matched first to prefer specific matches over generic ones (e.g., "Sodium Hyaluronate Crosspolymer" is matched before "Sodium").

				       Figure 9. Evidence Labelling Pipeline


If any ingredient in a product matches at least one evidence entry for concern c, that product receives label 1 for concern c; otherwise label 0. This produces a 7-dimensional binary label vector per product, which serves as the ground truth for DeBERTa training.
2.3.1.5 Worked example
Product: Anua Heartleaf Pore Control Cleansing Oil
Table 7 : INCI list (22 ingredients, scraped from Sephora HK)
Position i
Ingredient
Matched?
Mapped concerns
Weight w(i)
0
Ethylhexyl Palmitate
—
—
1.0000
1
Sorbeth-30 Tetraoleate
yes
redness, wrinkles
0.8963
2
Sorbitan Sesquioleate
—
—
0.8030
3
Caprylic/Capric Triglyceride
yes
acne, comedonal_acne, pores
0.7200
4
Butyl Avocadate
—
—
0.6450
5
Parfum/Fragrance
—
—
0.5781
6
Helianthus Annuus Seed Oil
—
—
0.5181
7
Macadamia Ternifolia Seed Oil
—
—
0.4643
8
Olea Europaea Fruit Oil
—
—
0.4161
9
Simmondsia Chinensis Seed Oil
—
—
0.3729
10
Vitis Vinifera Seed Oil
—
—
0.3342
11
Caprylyl Glycol
yes
redness, wrinkles
0.2998
12
Ethylhexylglycerin
yes
redness, wrinkles
0.2687
13
Curcuma Longa Root Extract
—
—
0.2408
14
Melia Azadirachta Flower Extract
—
—
0.2158
15
Tocopherol
yes
pigmentation
0.1934
16
Melia Azadirachta Leaf Extract
—
—
0.1734
17
Houttuynia Cordata Extract
yes
pigmentation, redness
0.1554
18
Corallina Officinalis Extract
—
—
0.1393
19
Melia Azadirachta Bark Extract
—
—
0.1248
20
Moringa Oleifera Seed Oil
—
—
0.1119
21
Ocimum Sanctum Leaf Extract
—
—
0.1000

Step-by-step evidence score calculation
Step 1 — Match. Six of 22 ingredients match against the evidence lookup.
Step 2 — Map to concerns. Each matched ingredient maps to one or more concerns 


Table 8 : Ingredient Matching with Concern
Matched ingredient
Position
Weight
Concerns
Sorbeth-30 Tetraoleate
1
0.8963
redness, wrinkles
Caprylic/Capric Triglyceride
3
0.7200
acne, comedonal_acne, pores
Caprylyl Glycol
11
0.2998
redness, wrinkles
Ethylhexylglycerin
12
0.2687
redness, wrinkles
Tocopherol
15
0.1934
pigmentation
Houttuynia Cordata Extract
17
0.1554
pigmentation, redness

Step 3 — Per-concern max. For each concern, take the maximum weight among all contributing ingredients. Interpretation. Redness and wrinkles receive the highest evidence






 


Table 9 : Per Concern Max
Concern
Contributing ingredients (weight)
scorec=max
Binary label
acne
Caprylic/Capric Triglyceride (0.720)
0.720
1
comedonal_acne
Caprylic/Capric Triglyceride (0.720)
0.720
1
pigmentation
Tocopherol (0.193), Houttuynia Cordata (0.155)
0.193
1
acne_scars_texture
(none)
0.000
0
pores
Caprylic/Capric Triglyceride (0.720)
0.720
1
redness
Sorbeth-30 (0.896), Caprylyl Glycol (0.300), Ethylhexylglycerin (0.269), Houttuynia Cordata (0.155)
0.896
1
wrinkles
Sorbeth-30 (0.896), Caprylyl Glycol (0.300), Ethylhexylglycerin (0.269)
0.896
1

scores (0.896) because the highest-ranked matched ingredient (Sorbeth-30 Tetraoleate, position 1) maps to those concerns. Pigmentation receives a lower score (0.193) because its only contributing ingredients appear late in the list (positions 15 and 17), suggesting low concentration. Acne_scars_texture receives 0 because no matched ingredient maps to that concern.

 		        Figure 10. Complete Flow of INCI INgredient map to Evidence Score
						






2.3.1.6 Multi-Label Classification (DeBERTa-v3-base)
DeBERTa (Decoding-enhanced BERT with disentangled attention) [28] and DeBERTa-v3 [27] introduce architectural innovations over the original BERT encoder [33]:
Disentangled attention. Each token is represented by two vectors  one for content and one for position  and attention scores are computed as a sum of content-to-content, content-to-position, and position-to-content terms. This allows the model to separately reason about what a token means and where it appears, which improves sensitivity to local ordering  critical when ingredient list position encodes concentration.
Enhanced mask decoder. An additional decoder layer incorporates absolute position information after all transformer layers, providing a richer pre-training signal without polluting the attention layers with absolute offsets.
DeBERTa-v3 further replaces masked language modelling with Replaced Token Detection (RTD, from ELECTRA), which is more sample-efficient during pre-training and yields stronger representations on small downstream datasets [27], [29].
2.3.1.6.1 Why not Bag-of-Words or BERT?
Table 9 below compares three candidate approaches for the multi-label ingredient classification task. The comparison is motivated by the characteristics of ingredient lists: (i) they are enumerations of technical chemical names, not natural sentences; (ii) ordering carries pharmacological meaning (INCI concentration ordering); and (iii) the dataset is small (1,009 products).










Table 10 : Model selection comparison (ingredient multi-label classification)
Property
Bag-of-Words (TF-IDF + classifier)
BERT-base-uncased
DeBERTa-v3-base (selected)
Input representation
Sparse term-frequency vector; no sub-word tokenization
WordPiece sub-word tokens; single embedding per token
SentencePiece tokens; separate content and position embeddings
Positional sensitivity
None — order-invariant by construction
Absolute positional embeddings (sinusoidal or learned)
Disentangled relative position; content-to-position cross-terms
Why it matters here
Cannot exploit INCI concentration ordering (ingredient position = importance); treats "Niacinamide, Water" identically to "Water, Niacinamide"
Captures order via absolute embeddings, but cannot directly compare the relative distance between two ingredient tokens
Relative position terms let the model learn that an ingredient appearing earlier than another is likely more concentrated, without memorizing absolute list lengths
Sub-word handling of chemical names
Exact n-gram matching only; novel names are out-of-vocabulary
WordPiece may split chemical names unpredictably (e.g., "Niacinamide" → "Ni", "##aci", "##na", "##mide")
SentencePiece with larger vocabulary (128K vs 30K); fewer fragmentation artifacts on chemical nomenclature
Pre-training objective
N/A (no pre-training)
Masked Language Modelling (MLM) — 15% random masking
Replaced Token Detection (RTD) — discriminative, more sample-efficient; better for small fine-tuning sets
Performance on small datasets
Competitive baseline but plateaus quickly; no transfer learning
Strong, but MLM pre-training is less sample-efficient than RTD
RTD pre-training yields stronger representations on datasets with < 5K samples [27], [29]
Parameters
Depends on vocabulary + classifier
~110M
~86M (v3-base, smaller due to efficient embedding sharing)

A bag-of-words baseline ignores ingredient ordering entirely, discarding the INCI concentration signal that the system relies on for position weighting. BERT captures ordering but through absolute positional embeddings, which are less effective for variable-length enumerations [33]. DeBERTa's disentangled relative positions and RTD pre-training provide a better inductive bias for this task, especially given the small corpus (1,009 products), where sample-efficient pre-training matters most [27]-[29]. This choice is also consistent with prior ingredient-aware beauty recommendation work, which treats ordered ingredient lists as meaningful sequential input rather than as unordered bags of tokens [1], [7].
To validate and complement the rule-based ingredient matching, a DeBERTa-v3-base transformer is trained for multi-label classification over the same 7 concern labels.Input representation. Each product's ingredient list is concatenated using [SEP] tokens as delimiters (e.g., [SEP] Niacinamide [SEP] Salicylic Acid [SEP] ...) and truncated to 512 tokens.


The ground truth for both the ingredient evidence scores and the DeBERTa multi-label classifier is the INCI-based evidence lookup, not marketing claims or manual annotations. Products are first labeled using the rule-based INCIDecoder lookup: each ingredient is matched against the evidence table, and binary labels (0/1) and position-weighted evidence scores are assigned per concern. These INCI-derived labels replace any marketing-based labels and serve as the training targets for DeBERTa. This keeps the system grounded in dermatological evidence rather than product claims [1].
2.3.1.7 ​​Labeling pipeline overview
The ground truth used to train and evaluate the DeBERTa classifier is derived entirely from the INCI-based evidence lookup (concern_lookup.json, sourced from INCIDecoder)  not from marketing claims or manual annotation. The pipeline proceeds as follows:
Ingredient matching. For each of the 1,009 products, each ingredient in its INCI list is matched against the evidence lookup (1,362 reference ingredients) using a three-tier strategy: exact match after normalization, forward substring, and reverse substring (longer evidence names matched first).
Binary labeling. If any ingredient in a product matches at least one evidence entry for concern c, that product receives label 1 for concern c; otherwise 0. This produces a 7-dimensional binary label vector per product.
Continuous evidence scoring. In parallel, each matched ingredient receives an INCI position weight. The product's evidence score for concern c is the maximum position weight among all ingredients matched to that concern. The binary labels (step 2) are used as DeBERTa training targets; the continuous scores (step 3) are used downstream for product ranking.
Table 11 : Multi-label dataset statistics (1,009 products, 7 concerns)
Concern
Positive (label = 1)
Negative (label = 0)
Prevalence (%)
acne
765
244
75.8
comedonal_acne
794
215
78.7
pigmentation
761
248
75.4
acne_scars_texture
393
616
38.9
pores
801
208
79.4
redness
864
145
85.6
wrinkles
852
157
84.4

Average labels per product: 5.18 / 7. Products with zero evidence labels: 103 (10.2%).
Table 12 : Label co-occurrence matrix


acne
com_acne
pigment
scars
pores
redness
wrinkles
acne
765
765
679
364
751
744
742
com_acne
765
794
699
393
780
769
765
pigment
679
699
761
352
704
742
737
scars
364
393
352
393
393
386
382
pores
751
780
704
393
801
774
770
redness
744
769
742
386
774
864
848
wrinkles
742
765
737
382
770
848
852

The matrix shows strong co-occurrence among most concerns (e.g., acne and comedonal_acne always co-occur: all 765 acne-positive products are also comedonal_acne-positive), reflecting the overlapping ingredient functions in the INCIDecoder lookup. acne_scars_texture is the most isolated label with the lowest prevalence (38.9%), which contributes to its lower classification performance
2.3.1.8 Input representation
Each product's ingredient list is concatenated using [SEP] tokens as delimiters and truncated to 512 tokens:
[SEP] Water [SEP] Niacinamide [SEP] Salicylic Acid [SEP] Zinc PCA [SEP] ...
This format ensures that the tokenizer's special-token handling separates ingredient boundaries while DeBERTa's disentangled attention can leverage relative positions to approximate concentration ordering.


2.3.1.9 Evaluation methodology
Evaluation uses 4-fold stratified cross-validation via MultilabelStratifiedKFold, which preserves the joint label distribution across folds [30], [31]. Each fold trains for 3 epochs with early stopping guided by micro F1.
At evaluation, per-label decision thresholds are optimized by sweeping 19 values from 0.05 to 0.95 and selecting the threshold that maximizes per-label F1 on the validation fold. This is necessary because class prevalence varies from 38.9% (acne_scars_texture) to 85.6% (redness), so a uniform 0.5 threshold would be suboptimal.
Metrics reported for this evaluation are : 
Label accuracy: fraction of individual (product, concern) predictions that are correct.
Subset accuracy (exact match): fraction of products where all 7 predicted labels exactly match all 7 ground truth labels.
Micro precision / recall / F1: computed over all (product × concern) predictions pooled together.
Macro F1: unweighted mean of per-label F1 scores.








The DeBERTa classifier outputs a 7-dimensional logit vector for each product — one raw score per concern. These logits are converted to independent probabilities via the sigmoid function (not softmax, since labels are not mutually exclusive — a product can address multiple concerns simultaneously):
pc=(logitc)=11+e-logitc


Each probability is then compared against a per-label decision threshold c to produce a binary prediction. The use of independent sigmoid outputs follows the standard multi-label classification setup, where each label is predicted independently rather than competing under a single softmax normalization.
 
Figure 11. Formula for Per-Label Decision Threshold
The thresholds c are tuned independently per label (see Threshold tuning above), producing a 7-dimensional binary prediction vector =[1,...,y7]  that is compared against the ground truth y=[y1,....,y7].
How metrics are computed from the 7-dimensional output. Given a validation fold with N products.
Table 13: the evaluation operates over an N x 7 prediction matrix
Metric
Computation
Scope
Label accuracy
Fraction of all N x 7 individual cells where c=yc
Per-cell
Subset accuracy
Fraction of the N products where i = yi for all 7 labels
Per-row (strict)
Micro F1
Precision, recall, and F1 pooled over all Nx7 predictions
Global pool
Macro F1
Mean of the 7 per-label F1 scores (each computed over Nproducts)
Per-column, averaged

In the 4-fold cross-validation, each of the 4 folds produces one set of these metrics. The reported values are the mean ± standard deviation across the 4 folds.

Figure 12. High Level Flow for Ingredient Analyzer and Evaluation Metrics




Table 14: DeBERTa-v3-base Model Architecture
Parameter
Value
Base model
microsoft/deberta-v3-base
Hidden size
768
Transformer layers
12
Attention heads
12
Vocabulary size
128,100
Max sequence length
512
Output
7 concern logits (multi-label)



Table 15: DeBERTa Training Configuration
Parameter
Value
Optimizer
AdamW [32]
Learning rate
2 x 10-5
Scheduler
Cosine with warmup ratio 0.06
Weight decay
0.01
Max gradient norm
1.0
Train batch size
8
Gradient accumulation steps
2 (effective batch = 16)
Epochs
3
Cross-validation
4-fold (MultilabelStratifiedKFold)
Best model selection
Micro F1
Dataset size
1,009 products









Loss function. Binary cross-entropy with logits and per-label positive class weighting to address class imbalance:
LBCE = -1Cc=17[wc+.yclog(yc)+(1-yc)log(1-log(yc))] 
where σ is the sigmoid function and the positive weight for each label is computed from the training fold:
wc+=ncnegncpos
This upweights the minority positive class for concerns with few positive products. The BCE-with-logits objective is used because the task is multi-label rather than multi-class, and AdamW is used as the optimizer following standard transformer fine-tuning practice [32]. Threshold tuning is defined in this work as an evaluation-time calibration step: per-label decision thresholds are optimized by sweeping 19 values from 0.05 to 0.95 and selecting the threshold that maximizes per-label F1. This accommodates the varying class prevalences across concerns.



			   Figure 12. DeBERTa Multi-Label Classification Flow












2.3.2 Review Analyzer
2.3.2.1 Dataset
The review corpus consists of 5,965 user reviews scraped from Sephora, covering 872 products in the catalogue. 
Table 16: review record 
Field
Description
review_text
The full body text of the review
headline
An optional short summary written by the reviewer
rating
Star rating on a 1–5 scale
product_url
URL linking the review to its product in the catalogue

Both review_text and headline are concatenated into a single text string for analysis, maximizing the amount of signal available from each review.
The 5,965 reviews are distributed across 872 products, yielding a mean of approximately 6.8 reviews per product. The distribution is right-skewed: a small number of popular products accumulate many reviews, while the majority of products have only a handful. Products with very few reviews naturally yield less reliable aggregate scores; when no reviews mention a particular concern, the system falls back to ingredient evidence alone (see Product Evidence Vector).

    		Figure 13. Distribution of Reviews


2.3.2.2 Approach
Instead of training a neural model on the relatively small and noisy review corpus, we adopt a keyword-based concern detection with context-aware sentiment classification approach. This avoids the need for labeled review data while maintaining interpretability, while also fitting the known challenges of aspect-level sentiment extraction in noisy review text [5]. The pipeline has three stages:
Stage 1: Concern Detection. 
For each review, keyword matching identifies which of the 7 concerns are mentioned. Each concern has a curated keyword list (e.g., acne: "acne", "pimple", "breakout", "blemish", "zit", "cystic acne"), supplemented by regex patterns for context-dependent mentions (e.g.,brighten\w*\s+(?:my\s+)?(?:skin|face|dark) for pigmentation).

Table 17: Complete concern keyword inventory
Concern
Keywords
Count
acne
acne, pimple, pimples, breakout, breakouts, break out, breaking out, broke out, blemish, blemishes, zit, zits, cystic acne
13
comedonal_acne
blackhead, blackheads, whitehead, whiteheads, comedone, comedones, comedonal, clogged pore, clogged pores
9
pigmentation
dark spot, dark spots, hyperpigmentation, pigmentation, discoloration, discolouration, melasma, sun spot, sun spots, uneven tone, uneven skin tone, dark mark, dark marks
13
acne_scars_texture
acne scar, acne scars, scarring, uneven texture, skin texture, rough skin, roughness, bumpy skin, ice pick
9
pores
my pore, my pores, large pore, enlarged pore, visible pore, open pore, minimize pore, minimise pore, refine pore, tighten pore, shrink pore, clogged pore, clogged pores
13
redness
redness, rosacea, my skin red, face red, inflamed, inflammation
6
wrinkles
wrinkle, wrinkles, fine line, fine lines, anti-aging, anti aging, antiaging, crow feet, crow's feet, laugh line, laugh lines, sagging
12


Keywords are curated to capture the colloquial language reviewers use (e.g., "zit" and "cystic acne" alongside the clinical term "acne"). Multi-word phrases (e.g., "clogged pore", "dark spot") are included to reduce false positives from single-word matches.
For concerns where simple keywords may miss contextual mentions, supplementary regex patterns are applied after keyword matching. These patterns capture discussion of a concern without using the concern's canonical keywords, for example, a reviewer saying "it brightened my skin" is discussing pigmentation even though the word "pigmentation" never appears.




Table 18: Supplementary concern regex patterns
Concern
Regex pattern
What it captures
pigmentation
brighten\w*\s+(?:my\s+)?(?:skin|face|complexion|dark)
"brightened my skin", "brightens dark complexion"
acne_scars_texture
(?:my|skin)\s+(?:texture|scars?)
"my texture", "skin scars"
acne_scars_texture
(?:smooth|soften)\w*\s+(?:my\s+)?(?:skin|face)
"smoothed my skin", "softens face"
pores
(?:my|visible|large|big|huge|open|clogged|enlarged)\s+pores?
"my large pores", "visible pores"
pores
pores?\s+(?:look|appear|are|seem|feel)\s+(?:smaller|bigger|larger|cleaner|clearer|tighter)
"pores look smaller", "pores feel tighter"
redness
(?:calm|soothe|reduc)\w*\s+(?:my\s+)?(?:redness|irritation|inflammation)
"calmed my redness", "reduces irritation"
redness
(?:skin|face)\s+(?:got|became|turn(?:ed)?|is|was)\s+red
"my skin turned red", "face got red"
redness
(?:irritat|sting|burn)\w+\s+(?:my\s+)?(?:skin|face)
"irritated my skin", "stinging my face"
wrinkles
(?:firm|plump|tighten)\w*\s+(?:my\s+)?(?:skin|face)
"firms my skin", "plumped my face"
wrinkles
(?:aging|ageing)\s+(?:skin|concern|sign)
"aging skin", "signs of ageing"

There are 10 supplementary regex patterns across 5 concerns. Acne and comedonal_acne do not require supplementary patterns because their keyword lists are already sufficiently distinctive — reviewers almost always use explicit terms like "breakout" or "blackhead" when discussing these concerns.
The output of Stage 1 is a set of detected concerns for the review (a subset of the 7 possible concerns). Concerns not mentioned receive a label of 0 and are not processed further.


Stage 2: Sentiment Classification. 
For each detected concern, context-aware regex patterns classify the sentiment as positive (improvement) or negative (worsening). Sentiment words and concern keywords rarely appear adjacent in natural text. To handle intervening words, all templates use a word gap fragment:
      w=(?:s+S+)0,5s+
This regex matches 0 to 5 intervening words between two terms. For example, the template help(s|ed)? W {kw} matches:
"helped my acne" (1 intervening word)
"helped clear up my acne" (3 intervening words)
"helped with my stubborn cystic acne" (4 intervening words)
but would not match "helped me feel more confident about my skin and cleared my acne" (too many words between "helped" and "acne"), avoiding false positives from long-range co-occurrences.


2.3.2.2.1 Positive sentiment templates (18 patterns)
Each template contains a {kw} placeholder that is instantiated with every keyword from the detected concern's keyword list. The ... below represents the word gap W.
Table 20: Positive sentiment templates
#
Template
Example match
1
help(s|ed)? ... {kw}
"helped my acne"
2
help(s|ed)?\s+(?:with\s+)?(?:my\s+)?{kw}
"helps with my breakouts"
3
reduc(e[ds]?|ing) ... {kw}
"reduced my redness"
4
clear(s|ed|ing)?\s+(?:up\s+)? ... {kw}
"cleared up my acne"
5
(?:got|get|getting)\s+rid\s+of ... {kw}
"got rid of dark spots"
6
improv(e[ds]?|ing) ... {kw}
"improved my wrinkles"
7
diminish(es|ed|ing)? ... {kw}
"diminished my fine lines"
8
(?:less|fewer|no\s+more|minimize[ds]?)\s+{kw}
"less redness", "no more breakouts"
9
{kw} ... (?:went\s+away|disappeared|cleared|improved|gone|reduced|faded)
"acne cleared", "dark spots faded"
10
(?:fad(e[ds]?|ing)|lighten(s|ed|ing)?) ... {kw}
"fading dark spots"
11
(?:great|good|amazing|excellent|perfect|fantastic|love|wonderful)\s+for\s+{kw}
"great for acne"
12
no\s+(?:more\s+)?{kw}
"no more pimples"
13
(?:prevent|prevents|prevented|preventing) ... {kw}
"prevented breakouts"
14
{kw}\s+(?:are|is|was|were)\s+(?:much\s+)?(?:better|less|smaller|fewer|gone)
"pores are smaller"
15
(?:healed?|healing) ... {kw}
"healed my blemishes"
16
(?:fight|fights|combat|combats) ... {kw}
"fights acne"
17
(?:tighten|minimiz|shrink|refin)\w* ... {kw}
"minimized my pores"
18
(?:smooth|soften)\w* ... {kw}
"smoothed my rough skin"



2.3.2.2.2 Negative sentiment templates (7 patterns)
Table 21: Negative sentiment templates
#
Template
Example match
1
(?:caus|gave|give|trigger)\w* ... {kw}
"caused breakouts"
2
(?:made|make|making) ... {kw}\s+worse
"made my acne worse"
3
(?:more|increased?|worsen)\s+{kw}
"more redness"
4
{kw}\s+(?:got|became|become)\s+worse
"acne got worse"
5
{kw}\s+(?:increased|appeared|flared|worsened)
"redness flared"
6
(?:didn.t|did\s+not|doesn.t|does\s+not|won.t)\s+help ... {kw}
"didn't help my acne"
7
no\s+(?:effect|improvement|change|difference) ... {kw}
"no improvement in dark spots"



Each template is instantiated with every keyword for the detected concern, and matching counts are tallied.
For each detected concern, the system iterates over all of that concern's keywords. For each keyword present in the text, it tests every positive and negative template (substituting the keyword for {kw}), counting the number of matches:

Figure 14. posscore

Figure 15. negscore
where Kcis the keyword set for concern c, and T+T-are the positive and negative template sets. The template-counting equations are defined in this work as a transparent scoring rule for converting review text into concern-level polarity counts.

Figure 16. Label based on score
The phrase "broke out" / "breaking out" / "break out" is a strong colloquial signal for a negative acne experience. The system applies a post-hoc override:
If the text matches (?:broke|break|breaking)\s+(?:me\s+)?out, the acne label is forced to −1.
Exception: if the match is preceded (within 3 words) by a negation word — stop, no more, prevent, didn't, doesn't, does not, don't, do not, without, never, won't, will not, hasn't, not — the override is suppressed (e.g., "stopped breaking out" remains positive).
The negation check regex:
(?:stop|no\s+more|prevent|didn.t|doesn.t|does\s+not|don.t|do\s+not
|without|never|won.t|will\s+not|hasn.t|not)\s+(?:\S+\s+){0,3}
(?:break|broke|breaking)\s+(?:me\s+)?out
Stage 3: Tie-breaking. 
When a concern is mentioned but the positive and negative pattern counts are equal or none, the star rating is chosen to resolve the ambiguity. This tie-breaking rule is also defined in this work and is used to avoid leaving detected concerns without a direction when explicit positive and negative template evidence is balanced.


Table 22: Star Rating for Ambiguity
Star rating
Condition
Assigned label
★★★★ or ★★★★★
rating ≥ 4
+1 (positive)
★ or ★★
rating ≤ 2
−1 (negative)
★★★
rating = 3
0 (neutral)

If a review's default rating is missing, it defaults to 3 (neutral).
A second tie-breaking pass also applies: after all concern labels and the "broke out" override are computed, any concern that was detected (mentioned in Stage 1) but still has a label of 0 gets a final rating-based assignment using the same rules above. This ensures that a review mentioning a concern is rarely left without a directional signal.

Figure 17. Review Analyzer Pipeline


Worked Examples
6.1 Example 1 — Positive review (straightforward)
Review input:
Field
Value
review_text
"This serum really helped clear up my acne and reduced my dark spots significantly. My pores look smaller too!"
rating
5
product
Example Serum X

Stage 1 — Concern Detection:
Match type
Text matched
Detected concern
Keyword
"acne"
acne
Keyword
"dark spots"
pigmentation
Regex
"pores look smaller" → pores?\s+(?:look|appear)\s+(?:smaller|...)
pores

Detected concerns: {acne, pigmentation, pores}
Stage 2 — Sentiment Classification:
Concern
Matched text
Template matched
pos
neg
Label
acne
"helped clear up my acne"
help(s|ed)? ... {kw} (positive #1)
1
0
+1
pigmentation
"reduced my dark spots"
reduc(e[ds]?|ing) ... {kw} (positive #3)
1
0
+1
pores
"pores look smaller"
{kw} ... (?:better|less|smaller|...) (positive #14)
1
0
+1

Stage 3 — Tie-Breaking: Not needed; all concerns had pos > neg.
Output label vector:
acne
comedonal_acne
pigmentation
acne_scars_texture
pores
redness
wrinkles
+1
0
+1
0
+1
0
0

6.2 Example 2 — Mixed review with "broke out" override and tie-breaking
Review input:
Field
Value
review_text
"I used this for my acne but it broke me out even more. At least my dark spots faded a bit."
rating
2

Stage 1 — Concern Detection:
Match type
Text matched
Detected concern
Keyword
"acne"
acne
Keyword
"broke out"
acne (already detected)
Keyword
"dark spots"
pigmentation

Detected concerns: {acne, pigmentation}
Stage 2 — Sentiment Classification:
Concern
Analysis
pos
neg
Label
acne
No positive templates match. No standard negative templates match either.
0
0
0 (tie)
pigmentation
"dark spots faded" matches {kw} ... (faded) (positive #9)
1
0
+1

"Broke out" override: "broke me out" matches the override pattern. Negation check: no negation word precedes it. → acne label is overridden to −1.
Stage 3 — Tie-Breaking: Acne already resolved by override. No remaining ties.
Output label vector:
acne
comedonal_acne
pigmentation
acne_scars_texture
pores
redness
wrinkles
−1
0
+1
0
0
0
0

6.3 Example 3 — Tie-breaking in action
Review input:
Field
Value
review_text
"I bought this for my wrinkles. It's okay I guess."
rating
4

Stage 1: Keyword "wrinkles" → detected concern: {wrinkles}
Stage 2: No positive or negative templates match (the reviewer mentions wrinkles but expresses no clear improvement or worsening). pos=0, neg=0 → tie.
Stage 3: Rating = 4 (≥ 4) → label = +1 (positive).
Output: wrinkles = +1, all others = 0.



7. Skin Type Suitability Detection
Skin type suitability is detected in parallel with concern sentiment using an analogous three-step approach. The system identifies 5 skin types: dry, oily, sensitive, normal, and combination.
7.1 Step 1 — Claim detection
Regex patterns identify whether the reviewer declares their skin type. Each skin type has 4–6 claim patterns covering common self-identification phrasings.
Table 5: Skin type claim patterns (representative examples)
Skin type
Example patterns
Matches
dry
(?:i\s+have|my)\s+(?:\w+\s+){0,2}dry\s+skin
"I have very dry skin", "my dry skin"
dry
my\s+skin\s+(?:is|was|tends?\s+to\s+be)\s+(?:very\s+)?dry
"my skin tends to be dry"
oily
(?:i\s+have|my)\s+(?:\w+\s+){0,2}oily\s+skin
"I have oily skin"
oily
oily[\s/-]+(?:to[\s/-]+)?(?:combo|combination)\s+skin
"oily to combination skin"
sensitive
(?:as\s+)?(?:a\s+)?(?:someone|person)\s+with\s+sensitive\s+skin
"as someone with sensitive skin"
sensitive
my\s+sensitive\s+skin
"my sensitive skin"
combination
(?:i\s+have|my)\s+(?:\w+\s+){0,2}(?:combination|combo)\s+skin
"I have combo skin"
normal
normal[\s/-]+(?:to[\s/-]+)?(?:dry|oily)\s+skin
"normal to dry skin"

7.2 Step 2 — Suitability scoring
For each claimed skin type, positive and negative templates (with {st} substituted for the skin type name) classify whether the product was suitable:
Positive suitability patterns (5 templates):
Template
Example match
(?:great|good|perfect|amazing|love|wonderful|ideal|best)\s+(?:for|on)\s+(?:my\s+)?{st}\s+skin
"perfect for my oily skin"
(?:works?|worked)\s+(?:really\s+)?(?:well|great|perfectly)\s+(?:for|on)\s+(?:my\s+)?{st}\s+skin
"works great on my dry skin"
(?:suitable|recommend(?:ed)?)\s+for\s+{st}\s+skin
"recommended for sensitive skin"
{st}\s+skin\s+(?:loves?|approved)
"oily skin approved"
(?:my\s+)?{st}\s+skin\s+(?:looks?|feels?)\s+(?:amazing|great|better|hydrated|...)
"my dry skin feels hydrated"

Negative suitability patterns (6 templates):
Template
Example match
(?:not\s+(?:good|great|suitable|ideal)|bad|terrible)\s+for\s+{st}\s+skin
"not good for oily skin"
(?:too\s+(?:dry|oily|heavy|greasy|rich|light))\s+for\s+(?:my\s+)?{st}\s+skin
"too heavy for my oily skin"
(?:doesn.t|does\s+not|didn.t|did\s+not)\s+work\s+(?:for|on)\s+(?:my\s+)?{st}\s+skin
"doesn't work on my sensitive skin"
(?:my\s+)?{st}\s+skin\s+(?:didn.t|doesn.t|does\s+not)\s+(?:like|tolerate|agree)
"my sensitive skin didn't tolerate it"
(?:not\s+suitable|not\s+recommend(?:ed)?)\s+for\s+{st}\s+skin
"not recommended for dry skin"
(?:harsh|irritating|drying|stripping)\s+(?:for|on)\s+(?:my\s+)?{st}\s+skin
"too drying for my dry skin"

7.3 Step 3 — Tie-breaking
The same star-rating fallback applies: when positive and negative suitability pattern counts are equal, rating ≥ 4 → +1, rating ≤ 2 → −1, rating = 3 → 0.
Aggregation
Individual review labels are aggregated per product. For each concern c, the product-level effectiveness score is:
 effc = nc+-nc-nc++nc-[-1,1]
Where nc+ and nc- are the counts of positive and negative reviews mentioning concern c. Products with no mentions for a concern receive a null score (excluded from the fusion step).




Review mention statistics
Table 6: Review mention statistics (5,965 reviews)
Concern
Positive
Negative
Total Mentions
Positive %
acne
255
141
396
64%
wrinkles
127
9
136
93%
redness
88
46
134
66%
pigmentation
101
6
107
94%
acne_scars_texture
101
6
107
94%
pores
59
9
68
87%
comedonal_acne
29
8
37
78%

The positive-to-negative ratio varies substantially across concerns. Acne has the most balanced distribution (64% positive), while pigmentation and acne_scars_texture are heavily skewed toward positive mentions (94%), reflecting the generally favorable outcomes reported by users for brightening and texture-improving products.














Table 9: Review Mention Statistics (5,965 reviews)




Concern
Positive
Negative
Total Mentions
acne
255
141
396
wrinkles
127
9
136
redness
88
46
134
pigmentation
101
6
107
acne_scars_texture
101
6
107
pores
59
9
68
comedonal_acne
29
8
37

The positive-to-negative ratio varies substantially across concerns. Acne has the most balanced distribution (64% positive), while pigmentation and wrinkles are heavily skewed toward positive mentions (>90%), reflecting the generally favorable outcomes reported by users for brightening and anti-aging products.







Product Evidence Vector
The ingredient evidence score and review effectiveness are fused into a single 7-dimensional product evidence vector:  




Where:
scing∈[0,1] is the INCI position-weighted ingredient evidence score
screv= effc +12∈[0,1] review effectiveness remapped from [−1,1] to [0,1]
The equal weighting (0.5/0.5) treats ingredient composition and real-world user outcomes as complementary evidence sources. When no reviews mention a particular concern, the product vector falls back to ingredient evidence alone, ensuring coverage even for products with sparse reviews.


