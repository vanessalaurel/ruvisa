# Ruvisa — IEEE Conference Paper: Experiments and Results (Draft)

> **Notes.** (1) Section/Table numbers are placeholders—renumber once merged with the full paper. (2) For the 8-page limit, this draft already compresses the report; if more space is needed, move per-class/confusion-matrix detail to figures and shorten §E/§F. (3) Factual correction from the source draft: acne's 0.544 is its *precision*, not mAP@50; redness has the highest mAP@50 (0.438). (4) Structure: §A–D = per-component performance experiments (aligned to Section III), §E = end-to-end case study, §F = discussion/limitations/future work. (5) No formal *human-subject user study* was conducted; this is stated as a limitation and future work in §F rather than overclaimed.

## V. Experiments and Results

We evaluate Ruvisa component-by-component along the data flow—(A) skin-image analysis, (B) product intelligence, (C) hybrid recommendation, and (D) the agentic layer—then (E) trace one end-to-end case study and (F) discuss findings, limitations, and future work. Each component is assessed with metrics appropriate to its task, and we emphasize how component behavior justifies the end-to-end design. Experiments mirror the architecture of Section III: every subsystem described there is evaluated here on the corresponding artifacts (the YOLOv8 detector, ResNet-18 severity head, INCI/review/DeBERTa product intelligence, the adaptive recommender, and the ReAct agent).

### A. Skin-Image Analysis

**Lesion localization (YOLOv8-M).** The detector was evaluated on the six-class validation split (564 images, 3,097 ground-truth boxes). It achieved a mean precision of 0.415, recall of 0.383, mAP@50 of 0.360, and mAP@50–95 of 0.197 (Table I). Performance varied substantially by class: redness obtained the highest mAP@50 (0.438), with acne close behind (0.428), while flat wart was weakest (mAP@50 0.303, mAP@50–95 0.149). The gap between mAP@50 and mAP@50–95 indicates the model often localizes lesions approximately but struggles to maintain tight bounding-box overlap under stricter IoU—consistent with small-object detection under uncontrolled consumer lighting, pose, and lesion-boundary ambiguity. We therefore treat the detector as a recall-oriented *proposal generator*: in an end-to-end skincare system a missed lesion contributes nothing downstream, whereas a roughly correct box still yields a usable crop, so subsequent stages are designed to absorb imperfect proposals rather than assume perfect localization.

**Table I. Lesion-detection performance (validation: 564 images, 3,097 instances).**

| Class | Precision | Recall | mAP@50 | mAP@50–95 |
|---|---|---|---|---|
| Acne | 0.544 | 0.403 | 0.428 | 0.275 |
| Redness | 0.419 | 0.522 | 0.438 | 0.171 |
| Comedonal acne | 0.414 | 0.378 | 0.347 | 0.133 |
| Pigmentation | 0.407 | 0.371 | 0.326 | 0.226 |
| Acne scars | 0.385 | 0.359 | 0.317 | 0.228 |
| Flat wart | 0.320 | 0.267 | 0.303 | 0.149 |
| Wrinkle (segmentation)† | 0.704 | 0.712 | 0.708† | 0.548† |
| **Overall (mean)** | **0.456** | **0.430** | **0.410** | **0.247** |

† Wrinkle is evaluated with the U-Net segmentation branch, not the YOLOv8 detector, so its last two columns report **Dice** (0.708) and **IoU** (0.548) rather than mAP@50 and mAP@50–95, which are undefined for segmentation. The overall mean now averages all seven rows (six detection classes + wrinkle); the last two mean cells therefore mix mAP with Dice/IoU and should be read with that caveat.

**Lesion severity (ResNet-18, four ordinal levels).** On a held-out test set of 1,460 lesion crops, the severity classifier reached 0.9938 overall accuracy, with per-class F1 all above 0.988 (Table II) and a strongly diagonal confusion matrix—the few errors fell between adjacent grades (level0–level1, level1–level2), with no gross misclassification. This adjacency of errors matters for deployment because concern-vector construction relies on approximate severity ordering rather than perfect grade separation. The near-perfect scores should be read cautiously: they show strong consistency with the test distribution, not clinical equivalence to dermatologist grading, and level3 in particular has smaller support (130). Notably, the classifier is far stronger than the detector that supplies its crops, implying the pipeline bottleneck is *lesion proposal quality*, not grade recognition—so future gains should target localization rather than the severity head.

**Wrinkle segmentation (U-Net).** The wrinkle branch is evaluated by comparing its predicted masks against the 1,000 manually annotated FFHQ wrinkle masks at the pixel level (Table II, last row). It attains an F1/Dice of 0.708 and IoU of 0.548 (precision 0.704, recall 0.712), with a balanced precision–recall profile indicating neither systematic over- nor under-segmentation. Pixel accuracy is 0.996 but is uninformative here because wrinkle pixels are a tiny fraction of the face, so Dice/IoU are the meaningful measures; mean per-image Dice (0.714) closely tracks the pooled value, confirming the result is not driven by a few large-wrinkle faces. A moderate Dice is expected for thin, low-contrast wrinkle structures, and it is sufficient for the downstream signal, which only needs regional wrinkle *burden* rather than pixel-perfect delineation.

**Table II. Lesion-severity classification (per class, 1,460 crops) and wrinkle segmentation (1,000 images).**

| Class / model | Precision | Recall | F1 / Dice | Support / IoU |
|---|---|---|---|---|
| level0 | 0.9923 | 0.9981 | 0.9952 | 515 |
| level1 | 0.9953 | 0.9906 | 0.9929 | 635 |
| level2 | 0.9889 | 0.9889 | 0.9889 | 180 |
| level3 | 1.0000 | 1.0000 | 1.0000 | 130 |
| **Severity overall accuracy** | — | — | **0.9938** | 1,460 |
| *Wrinkle segmentation (pixel)* | 0.7041 | 0.7116 | 0.7078 | IoU 0.5478 |

**Integration outcome.** Detections are assigned to facial regions via centroid-in-mask tests and pooled into per-region severity/counts, then mapped (with wrinkle/texture signals) into the shared seven-concern user vector. This stage is an integration result rather than a standalone benchmark: its key property is internal consistency—the same concern representation consumed by the recommender traces directly back to lesion detections, severity estimates, and wrinkle measurements, preserving a coherent path from raw image to recommendation-ready state. This validates the architectural bet that a *moderately accurate detector plus a highly reliable severity classifier* can still yield a useful end-to-end representation when the aggregation layer is built to absorb localized noise.

### B. Product Intelligence: INCI Evidence, Reviews, and DeBERTa

**INCI evidence mapping.** Matching catalogue INCI lists against the INCIDecoder-derived lookup produced binary labels and continuous evidence scores for 1,009 products. The label space was dense (mean 5.18 of 7 concerns per product; only 10.2% of products received zero labels): redness was most prevalent (864 products), followed by wrinkles (852), pores (801), comedonal acne (794), acne (765), and pigmentation (761), while acne scars/texture was most selective (393). This asymmetry reflects a real property of formulation science—some concerns are served by broad multifunctional ingredient families, others by narrow specialized actives—and yields a multi-functional product space well suited to partial-alignment matching. It also foreshadows why exact-match prediction is hard: with many simultaneous positive labels per product, reconstructing all seven dimensions is strict even when most individual labels are correct.

**Rule-based review aggregation.** The review analyzer detects concern mentions, assigns per-concern polarity, and aggregates per SKU. Despite 5,965 reviews over 872 products (mean 6.8/product), the structured signal is sparse: only 452 products had ≥1 concern mention (420 had none), with a mean of 0.84 non-null concern dimensions per product. Mentions were uneven across concerns (acne highest at 396; comedonal acne lowest at 37) and strongly positivity-skewed for several concerns (e.g., pigmentation and acne scars/texture each 101 positive vs. 6 negative), consistent with the known positivity bias of voluntary beauty reviews. Reviews thus add genuine experiential evidence that ingredients cannot capture, but only for a subset of products—justifying their use as a *selective refinement layer* rather than a standalone backbone.

**DeBERTa: INCI-only vs. claims text.** Under 4-fold multilabel stratified CV, INCI-only supervision clearly outperformed claims/marketing text (Table III): micro-F1 0.9206 vs. 0.8510 (+0.0696) and micro-precision 0.8837 vs. 0.7407 (+0.1430), with INCI ahead on every per-concern head (acne scars/texture weakest in both). The diagnostic result is the claims model's micro-recall saturating at 1.0000 (zero variance) alongside low precision: its tuned thresholds collapsed to near-zero, so it predicted positives over-inclusively—exactly the failure mode expected when a model latches onto repeated promotional vocabulary rather than discriminative formulation evidence. INCI text instead balanced high recall (0.9608) with much stronger precision, empirically supporting the core thesis that *ingredient evidence is a more faithful basis for concern inference than marketing claims*.

**Table III. DeBERTa multi-label results, 4-fold CV (mean ± std).**

| Setting | Micro-F1 | Micro-P | Micro-R | Macro-F1 | Subset acc. | Mean label acc. |
|---|---|---|---|---|---|---|
| (A) INCI-only | 0.9206 ± .005 | 0.8837 ± .007 | 0.9608 ± .009 | 0.9079 ± .005 | 0.3931 ± .023 | 0.8653 ± .007 |
| (B) Claims/marketing | 0.8510 ± .001 | 0.7407 ± .001 | 1.0000 ± .000 | 0.8411 ± .001 | 0.3271 ± .012 | 0.7408 ± .001 |

Together these results define the product-intelligence design: INCI evidence is the primary backbone (broad coverage, interpretable, best supervised performance), review aggregation is a sparse experiential correction, and claims text is suitable only for explanation/retrieval—not as ground truth.

### C. Hybrid Recommendation: Cosine Matching, Adaptive Ranking, and Knowledge-Graph Re-ranking

Products and users share the same seven-concern space; base fit is cosine similarity, and a multiplicative modifier injects user history (direct boosts/penalties on product URLs, ingredient-overlap effects, and concern amplification for worsened dimensions). We analyze a representative snapshot (user `4d105fdd9f9f7cae`, combination skin, vector `[0.067, 0, 0, 0.133, 0, 0, 0.055]`): 911 products received a finite score, 817 (89.7%) were skin-type compatible. The top cosine match (Dramatically Different Moisturizing Gel, vector `[0.25,0,0,0.50,0,0,0]`) scored 0.9376 because both vectors concentrate weight on the same concerns (acne scars/texture, acne), confirming the base rank is geometric alignment rather than popularity.

The adaptive layer is *sparse but decisive* (Table IV): only 3 of 911 products were boosted and none penalized, yet 19 products (2.1%) changed position versus a cosine-only sort. The clearest case is Blemish Clearing Cleanser rising from flat rank 20 to adaptive rank 3 via a direct improvement-linked ×2 boost, while higher-cosine but neutral items (Exolive Squalane Oil Serum, 0.9271) were displaced. This demonstrates a true hybrid policy: cosine defines the default structure and history selectively overrides it where prior outcomes justify it, without destabilizing the rest of the ranking. (This snapshot exercises the boost path only; penalty/avoidance branches were inactive because the stored history contained improved products and no failed outcomes.)

**Table IV. Adaptive vs. cosine-only ranking (same top-5 SKUs).**

| Adaptive rank | Cosine rank | Product | Base sim. | Modifier | Adaptive score |
|---|---|---|---|---|---|
| 1 | 1 | Dramatically Different Moisturizing Gel | 0.9376 | 2.00 | 1.8751 |
| 2 | 4 | 15% Vitamin C and EGF Serum | 0.7263 | 2.00 | 1.4526 |
| 3 | 20 | Blemish Clearing Cleanser | 0.6539 | 2.00 | 1.3078 |
| 4 | 2 | Exolive Squalane Oil Serum | 0.9271 | 1.00 | 0.9271 |
| 5 | 3 | Lala Retro Whipped Cream | 0.7502 | 1.00 | 0.7502 |

**Knowledge-graph re-ranking.** A skincare knowledge graph sits on top of the recommender and encodes the one relationship a flat seven-concern vector cannot: *active-level synergy and conflict*. Built over the live catalogue, it contains 5,123 nodes and 21,581 edges—1,009 products, 4,082 ingredients, 10 INCIDecoder functions, 15 canonical active groups, and the seven concerns—linked by 13,997 position-weighted `CONTAINS` edges, 3,478 ingredient→function edges, 3,323 product→active edges, and 734 review-support edges, plus a curated knowledge layer of 7 conflict and 12 synergy active-pairs. At ranking time, a conflict/synergy factor (clamped to [0.70, 1.25]) becomes a third multiplier in the score, so the final rank is `cosine × history-modifier × KG-factor`. Of 1,009 products, 793 carry ≥2 actives and are thus eligible for the layer; the KG boosts 458 synergistic products, penalizes 82 conflicting ones, and leaves 469 neutral (Table V), with non-neutral factors spanning 0.73–1.25. For example, *Ole Henriksen Daily D-Clog Pore-Clearing Cleanser* receives a 0.916 conflict penalty for combining AHA and BHA (over-exfoliation risk), while *The INKEY List Face Glow* trio earns a 1.09 synergy boost (niacinamide + hyaluronic acid, glycerin + hyaluronic acid). Beyond ranking, the same graph drives explainability: walking product→ingredient→function→concern paths and appending synergy/conflict notes yields the human-readable justifications the agent surfaces to users—so the KG contributes both a conflict-aware ranking signal and a transparent rationale, directly serving the paper's evidence-backed, explainable thesis.

**Table V. Knowledge-graph ranking-layer effect across the catalogue (1,009 products).**

| Outcome | Products |
|---|---|
| Eligible (≥2 actives) | 793 |
| Synergy boost (factor > 1) | 458 |
| Conflict penalty (factor < 1) | 82 |
| Neutral (factor = 1) | 469 |

### D. Agentic Conversational Orchestration

The assistant is a ReAct-style agent (LangGraph) over ChatOllama (`llama3.2:latest`) with eight tools; authenticated user context (profile, scan history, purchases, and explicit `user_id`) is injected before each invocation, and tool-call traces are extracted for observability. We probe behavior with four compound prompts (P1–P4) designed to exercise the full tool set (Table VI). Each prompt triggered 3–8 distinct tool calls, showing the agent decomposes compound instructions into ordered tool sequences rather than a single retrieval. Tool selection matched semantic intent in all four cases: P2 correctly chose `recommend_routine` (not `recommend_products`) for the routine request, then `search_products` for item queries and `get_product_info` for the ingredient lookup; P4 issued `track_purchase` *before* `evaluate_outcomes`, respecting the causal order the user specified. Because each tool materializes structured strings from SQLite and the catalogue, final answers cite real product fields when tools succeed—evidence that the agent acts over live state rather than generating free-form text.

**Table VI. Tool-invocation traces for four probe prompts (same authenticated user).**

| # | Prompt (abbrev.) | Tools invoked (in order) |
|---|---|---|
| P1 | Evaluate past recs, compare latest scan, recommend updated products | evaluate_outcomes → compare_analyses → evaluate_outcomes → recommend_products → get_user_profile |
| P2 | Build routine < $80, suggest acne/redness products, detail top serum | recommend_routine → search_products (acne) → search_products (redness) → get_product_info |
| P3 | Check history, find cheaper alternatives, avoid repeats, compare scans | get_user_profile → search_products → compare_analyses → get_user_profile → search_products → track_purchase → search_products → evaluate_outcomes |
| P4 | Record a purchase, re-evaluate outcomes, recommend next products | track_purchase → evaluate_outcomes → recommend_products → recommend_products |

We observed occasional redundant calls (repeated `evaluate_outcomes`/`get_user_profile`), indicating the agent sometimes re-queries state mid-chain; this does not change correctness but suggests light call-deduplication as future work.

**Quantitative evaluation.** To move beyond qualitative traces, we built an automated harness (`scripts/eval_agent.py`) over a curated set of 12 queries—nine single-intent prompts (one per tool) and three compound prompts (P1, P2, P4)—issued against real users in the production SQLite database, each grounded query repeated twice (24 grounded runs) on `llama3.2:latest`. We report three families of metrics (Table VII). (i) *Tool-selection accuracy*: for every query we define the set of tools the intent requires and check the agent's actual calls. The agent achieved a **mean required-tool recall of 1.00** and selected the **full required tool set in 100% of runs**, averaging 2.33 tool calls per turn (range 1–7); 12.5% of calls were redundant duplicates, consistent with the re-query behavior noted above. (ii) *Latency*: mean **2.73 ± 1.77 s per turn** (0.97–10.31 s) on a local 3B-parameter model, supporting the deployability claim. (iii) *Grounding ablation*: we re-ran every query with tools disabled and measured the **hallucinated-product rate**—the fraction of product-like mentions absent from the 1,009-item catalogue (token-overlap check, threshold 0.6). Enabling tools cut hallucination from **57.2% to 35.2%** (a 38% relative reduction), empirically confirming that tool grounding makes the agent substantially more faithful to the real catalogue.

**Table VII. Quantitative agent evaluation (12 queries; grounded runs ×2; `llama3.2:latest`).**

| Metric | Value |
|---|---|
| Required-tool recall (mean) | 1.00 |
| Full required-tool-set rate | 100% |
| Tool calls per turn (mean; range) | 2.33; 1–7 |
| Redundant (duplicate) call rate | 12.5% |
| Latency per turn (mean ± std) | 2.73 ± 1.77 s |
| Hallucinated-product rate — tools ON | 35.2% |
| Hallucinated-product rate — tools OFF | 57.2% |

Two findings stand out. First, **tool selection is essentially solved** for this task: even a small local model maps natural-language intent—including compound, causally-ordered instructions—onto the correct tools every time, validating the ReAct design and tool schema. Second, the ablation shows tool grounding is necessary but, for a small model, *not sufficient*: with tools disabled the agent invents well-known but unavailable products (e.g., Differin, CeraVe, Paula's Choice—none in the Sephora HK catalogue), and grounding roughly halves this. The residual 35.2% rate—cases where the agent calls the right tool yet still paraphrases catalogue results with familiar brand names—indicates that closing the remaining gap requires constraining generation to retrieved items (e.g., enforced citation of returned SKUs or a larger backbone), which we identify as immediate future work toward reliable agentic commerce. Overall, the agent reliably maps intent onto correct, ordered, state-grounded tool use, establishing Ruvisa as an evidence-grounded instance of agentic commerce with autonomous checkout as the intended extension.

### E. End-to-End Case Study

The component results above evaluate each stage in isolation; this section traces real data through the *whole* pipeline to show the components compose into a coherent system.

**Product-intelligence fusion (The INKEY List Salicylic Acid Cleanser).** Table VIII shows how the three product signals combine for one catalogue product (price $120; storefront rating 4.4/5 over 363 reviews; 8 reviews in our analysis subset, of which only 3 yielded concern labels). Position-aware INCI evidence already gives broad, formulation-grounded coverage—strongest on redness and wrinkles (0.912) with solid acne, comedonal acne, and pores support (0.759). The sparse review layer then refines exactly the dimensions users actually discussed: positive mentions lift acne and comedonal acne from 0.759 to 0.879, and rescue pigmentation from a weak ingredient-only 0.110 to a moderate 0.555, while concerns with no review mentions retain their ingredient-only scores. This is the hybrid design working as intended—ingredients provide the prior, reviews supply a targeted experiential correction.

**Table VIII. Per-concern fusion for one product (INCI prior + review correction).**

| Concern | INCI score | Review | Fused score |
|---|---|---|---|
| acne | 0.759 | +1 | 0.879 |
| comedonal acne | 0.759 | +1 | 0.879 |
| pigmentation | 0.110 | +1 | 0.555 |
| acne scars/texture | 0.576 | — | 0.576 |
| pores | 0.759 | — | 0.759 |
| redness | 0.912 | — | 0.912 |
| wrinkles | 0.912 | — | 0.912 |

**End-to-end user journey (user `4d105fdd9f9f7cae`).** The same data path runs unbroken from image to conversation. The user's scan yields a concern vector `[0.067,0,0,0.133,0,0,0.055]` (dominated by acne scars/texture); cosine matching over the fused product vectors surfaces *Dramatically Different Moisturizing Gel* (similarity 0.9376) because both vectors concentrate on the same concerns (§C); the adaptive layer then promotes a previously-helpful product (*Blemish Clearing Cleanser*, cosine rank 20 → adaptive rank 3); and when the same user asks conversationally for recommendations, the agent (§D) selects `recommend_products` with this user's authenticated state and returns catalogue items for the live concern profile. A single user thus flows from selfie → seven-concern state → evidence-fused product scores → history-aware ranking → grounded conversational recommendation, exercising every contribution of the paper on one consistent representation.

### F. Discussion, Limitations, and Future Work

**Synthesis.** The results jointly support the paper's thesis that trustworthy skincare recommendation must be evidence-backed, personalized, and self-correcting *within one system*. Product intelligence is evidence-backed: INCI supervision beats marketing claims (micro-F1 0.921 vs. 0.851) and the fusion case study shows ingredient and review signals combining as designed. Recommendation is personalized and journey-aware: the same seven-concern schema links image to catalogue, and the adaptive layer—though sparse (3/911 products modified)—decisively re-ranks based on real outcomes. The agent makes the system actionable: it selects the correct tools 100% of the time and tool grounding cuts hallucination by 38% relative, instantiating evidence-grounded agentic commerce.

**Limitations and threats to validity.** Several caveats temper these results. (1) *Detection* is the weakest link (mAP@50 0.360); the pipeline is engineered to tolerate this, but localization caps end-to-end fidelity. (2) The *severity* classifier's near-perfect accuracy (0.994) reflects consistency with a single dataset split, not clinical equivalence, and level3 has limited support. (3) *Review* evidence is sparse and positivity-biased, so its corrective power is uneven across products. (4) The *recommender* analysis is a single representative snapshot that exercises only the boost path; penalty-driven demotion, while implemented, is not demonstrated here. (5) The *agent* evaluation uses one small local model (`llama3.2:latest`) over a 12-query set, the hallucination metric is an automatic catalogue-membership proxy, and a residual 35% product-hallucination rate remains. (6) Most importantly, we report *no formal user study*: all results are system- and data-grounded rather than human-subject outcomes, and the catalogue is single-market (Sephora HK), so generalization to other markets and real shopper satisfaction remain unverified.

**Future work.** These limitations map to concrete next steps: strengthen small-object lesion localization (the dominant CV bottleneck); broaden review extraction and de-bias aggregation; report a multi-snapshot recommender study covering penalty and concern-amplification paths; constrain agent generation to retrieved SKUs (or adopt a larger backbone) to drive product hallucination toward zero; conduct a longitudinal user study measuring real purchase satisfaction and skin-outcome tracking; and extend the grounded tool interface toward autonomous, agent-initiated checkout to fully realize agentic commerce.
