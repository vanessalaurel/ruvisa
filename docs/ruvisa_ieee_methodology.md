# Ruvisa — IEEE Conference Paper: Methodology (Draft)

## III. Methodology

Ruvisa is an evidence-grounded decision-support system for personalized skincare product recommendation. Rather than treating skincare recommendation as a popularity ranking or an open-ended text-generation task, Ruvisa aligns visual skin evidence, product formulation evidence, review-derived experiential evidence, and user history into a shared concern representation. This enables the system to recommend products because their evidence profile matches the user's current skin state, not merely because they are highly rated or textually similar to a query.

### A. System Overview

Ruvisa consists of four main layers. The first layer is a *skin analyzer* that converts a user's facial image into a structured concern vector. The second layer is a *product analyzer* that converts INCI ingredient lists and user reviews into product evidence vectors. The third layer is an *adaptive recommendation module* that matches users and products through concern-space similarity and history-aware re-ranking, augmented by a *skincare knowledge graph* that makes ingredient–function–concern relationships explicit and encodes ingredient synergy and conflict. The fourth layer is a *tool-grounded conversational agent* that exposes functions for recommendation, explanation, product search, routine construction, and progress tracking through natural language.

All layers are connected by a unified seven-concern schema:

\[
C=\{\text{acne},\ \text{comedonal acne},\ \text{pigmentation},\ \text{acne scars/texture},\ \text{pores},\ \text{redness},\ \text{wrinkles}\}.
\]

This shared schema is central to Ruvisa. The skin analyzer does not output only isolated lesion detections; it projects visual evidence into concern-level scores. Similarly, the product analyzer does not output only textual descriptions; it projects ingredient and review evidence into concern-level product scores. As a result, the user's skin state and each product's evidence profile can be compared directly in the same vector space, and the knowledge graph can reason over the ingredients behind those scores.

Formally, a user is represented as \(u=(u_1,\ldots,u_7)^\top\), where \(u_c\in[0,1]\) denotes the severity or priority of concern \(c\). Each product \(j\) is represented as \(v_j=(v_{j,1},\ldots,v_{j,7})^\top\), where \(v_{j,c}\in[0,1]\) denotes the strength of evidence that product \(j\) is relevant to concern \(c\). Recommendation is therefore formulated as evidence-grounded matching between the current user state and product-level concern evidence.

### B. Skin Analyzer

The skin analyzer transforms a user facial image into the concern vector \(u\). It is implemented as a four-stage computer-vision pipeline: face parsing, lesion detection, lesion severity classification, and wrinkle segmentation.

First, a BiSeNet face parser isolates facial skin regions and constructs coarse anatomical areas such as the forehead, cheeks, and chin. Face parsing removes irrelevant background, hair, and accessories from downstream analysis, and provides region masks for assigning detected lesions to meaningful facial zones. This allows the system to report not only which concerns are present, but also where they appear.

Second, lesion localization is performed using YOLOv8m. The detector outputs bounding boxes, confidence scores, and lesion classes for visible skin findings such as acne, comedonal acne, pigmentation, redness, and acne scars. In Ruvisa, the detector is used as a proposal generator: its role is to identify candidate visual evidence for recommendation, rather than to produce a standalone clinical diagnosis.

Third, each detected lesion crop is passed to a ResNet-18 severity classifier. The classifier predicts one of four ordinal severity levels (none, mild, moderate, severe), which are mapped into normalized scalar values \(s_i\in\{0,\tfrac{1}{3},\tfrac{2}{3},1\}\), where \(s_i\) is the severity score of lesion \(i\). This step is important because a few severe lesions may imply a different product need from many mild detections.

Fourth, wrinkles are handled through a U-Net segmentation branch. Unlike acne-like lesions, wrinkles are thin and spatially distributed structures, so they are modeled as a segmentation problem rather than an object-detection problem. The wrinkle branch estimates a binary wrinkle mask and converts wrinkle coverage into the same \([0,1]\) range as other concerns:

\[
u_{\text{wrinkles}}=\min\!\left(1,\ \frac{\text{wrinkle\_pct}}{5}\right).
\]

For non-wrinkle concerns, detected lesion classes are mapped into the shared concern schema and then aggregated by count and severity. Let \(N_c\) be the number of detected lesions associated with concern \(c\), and let \(\bar{s}_c\) be their average severity. The count score is capped to reduce sensitivity to unusually dense detections, \(\text{count\_score}_c=\min\!\left(1,\ N_c/10\right)\), and the final concern score is:

\[
u_c=\text{count\_score}_c\,(0.5+0.5\,\bar{s}_c).
\]

This formulation increases the concern score when more lesions are observed, while also reflecting how severe those lesions are. The output of the skin analyzer is therefore a stable seven-dimensional user concern vector rather than a collection of unstructured visual predictions.

### C. Product Analyzer

The product analyzer constructs product evidence vectors from two complementary sources: formulation evidence from INCI ingredient lists and experiential evidence from user reviews. Its purpose is to convert heterogeneous product information into the same seven-concern schema used by the skin analyzer.

The product catalogue is scraped from Sephora Hong Kong and includes product title, brand, price, category, star rating, user reviews, and INCI ingredient lists. Products with valid INCI lists are processed by the formulation branch, while products with available reviews are additionally processed by the review branch. When review evidence is sparse or unavailable, Ruvisa falls back to ingredient evidence alone.

#### 1) Ingredient Evidence from INCI Lists

INCI ingredient lists are used as the primary product-evidence source because they describe what a product contains, rather than what it claims in marketing text. Since INCI order carries approximate concentration information, Ruvisa assigns position-dependent weights to ingredients. For an ingredient at position \(i\) in a list of length \(N\), the position weight is:

\[
w(i)=\max\!\left(0.1,\ e^{-2.3\frac{i}{N-1}}\right).
\]

Early ingredients therefore contribute stronger evidence, while later ingredients retain weak evidence through the floor value of 0.1.

To build the ingredient–concern knowledge base, ingredient functions are collected from INCIDecoder function pages such as anti-acne, skin-brightening, soothing, exfoliant, cell-communicating ingredient, astringent, and antioxidant. These functions are mapped onto the seven-concern schema. For example, anti-acne ingredients map to acne and comedonal acne, skin-brightening ingredients map to pigmentation, soothing ingredients map to redness, exfoliants map to pores and texture-related concerns, and cell-communicating ingredients map to wrinkles.

Each product ingredient list is matched against this lookup table using a three-tier strategy: exact normalized matching, forward substring matching, and reverse substring matching. Longer names are prioritized to reduce false matches from generic substrings. For each product \(j\) and concern \(c\), the ingredient evidence score is defined as the strongest position-weighted match:

\[
s^{\text{ing}}_{j,c}=\max_{i\in M_{j,c}} w(i),
\]

where \(M_{j,c}\) is the set of matched ingredients in product \(j\) associated with concern \(c\). The maximum is used instead of a sum because concern relevance is often driven by the most concentrated relevant active ingredient rather than by the number of weak matches.

In addition to this rule-based evidence mapping, Ruvisa uses a DeBERTa-v3-base multi-label classifier to learn concern labels from product text. The classifier is trained using ingredient-derived labels and serves as a learned generalizer for product understanding. This allows the system to compare structured INCI evidence with transformer-based text modeling while keeping ingredient-grounded supervision as the primary source of truth.

#### 2) Review-Derived Experiential Evidence

Ingredient lists describe formulation potential, but they do not capture how users experience a product. Ruvisa therefore uses reviews as an experiential refinement layer. Each review is normalized by combining the body text and headline when available. The system then detects concern mentions using curated keyword lists and regular expressions covering both formal and colloquial skincare language, such as "breakout", "dark spots", "large pores", "fine lines", and "irritation".

After concern detection, sentiment direction is classified for each mentioned concern. Positive templates capture improvement phrases such as "helped my acne", "reduced redness", or "pores look smaller". Negative templates capture worsening phrases such as "caused breakouts", "made my acne worse", or "no improvement". A special override treats "broke out" as a strong negative acne signal unless it is negated by phrases such as "no more breaking out".

For each review and concern, positive and negative pattern matches are counted. If positive matches exceed negative matches, the concern label is \(+1\); if negative matches exceed positive matches, the label is \(-1\). If no clear direction is found, the review star rating is used as a tie-breaker: four- and five-star reviews are treated as positive, one- and two-star reviews as negative, and three-star reviews as neutral.

Product-level review effectiveness is aggregated per concern:

\[
\text{eff}_{j,c}=\frac{n^+_{j,c}-n^-_{j,c}}{n^+_{j,c}+n^-_{j,c}+\epsilon},
\]

where \(n^+_{j,c}\) and \(n^-_{j,c}\) are the numbers of positive and negative review mentions for concern \(c\). This value is remapped into \([0,1]\):

\[
s^{\text{rev}}_{j,c}=\frac{\text{eff}_{j,c}+1}{2}.
\]

Reviews are also used to infer skin-type suitability for dry, oily, sensitive, normal, and combination skin by detecting self-declared skin-type statements and positive or negative suitability phrases.

#### 3) Product Evidence Fusion

The final product vector combines ingredient evidence and review-derived evidence. For each product \(j\) and concern \(c\), the product evidence score is:

\[
v_{j,c}=
\begin{cases}
0.5\,s^{\text{ing}}_{j,c}+0.5\,s^{\text{rev}}_{j,c}, & \text{if review evidence is available},\\[4pt]
s^{\text{ing}}_{j,c}, & \text{otherwise}.
\end{cases}
\]

This design treats INCI evidence as the primary source and review evidence as a complementary calibration signal. The fallback to ingredient evidence is important because reviews are sparse and unevenly distributed across products. The output of the product analyzer is a seven-dimensional evidence vector \(v_j\), which is passed to the recommendation layer.

### D. Recommendation and Adaptive Personalization

Given a user concern vector \(u\) and a product evidence vector \(v_j\), Ruvisa first computes base relevance using cosine similarity:

\[
\text{sim}(u,v_j)=\frac{u^\top v_j}{\lVert u\rVert\,\lVert v_j\rVert}.
\]

Cosine similarity is used because it compares the *direction* of concern emphasis rather than requiring the absolute scale of the skin analyzer and product analyzer to be identical. A product is ranked highly when its evidence profile aligns with the user's current skin-concern profile. Products with zero evidence or infeasible constraints, such as budget violations, can be filtered before ranking.

This base ranking is static: it reflects the user's current skin state and the product's evidence profile at a single point in time. Ruvisa extends it with adaptive personalization based on purchase history and follow-up skin scans.

When a user records a purchased product and later uploads a new skin analysis, the system compares concern-level changes between scans. If more concerns worsen than improve, the product is attributed a worsened outcome; if more concerns improve, an improved outcome; and if changes are small or balanced, a no-change or mixed status. This attribution is converted into a multiplicative modifier \(m_j\). Previously worsened, mixed, or ineffective products receive penalties, while improved products receive boosts, so the system learns from a user's own product history rather than repeatedly suggesting products with poor prior outcomes.

Ruvisa also generalizes feedback to unseen products through ingredient-overlap reasoning. If a candidate product has high ingredient overlap with a previously failed product, it is penalized; if it overlaps with a previously successful product, it may be boosted. Ingredient overlap is measured with Jaccard similarity:

\[
J(I_j,I_k)=\frac{|I_j\cap I_k|}{|I_j\cup I_k|},
\]

where \(I_j\) and \(I_k\) are evidence-matched active-ingredient sets.

Temporal concern boosting is applied when a concern worsens between scans. If concern \(c\) increases by \(\Delta_c>0.03\), the user vector is adjusted before similarity computation:

\[
u'_c=\min\!\left(1.0,\ u_c(1+3\Delta_c)\right).
\]

This shifts ranking toward products that address newly worsening concerns without discarding the original concern profile.

#### 1) Knowledge-Graph–Aware Conflict and Synergy Reasoning

A flat seven-concern vector captures *which* concerns a product addresses, but it cannot express how a product's active ingredients interact with one another. Ruvisa therefore layers a *skincare knowledge graph* on top of the recommender to make ingredient relationships explicit and to encode the one piece of domain knowledge a concern vector cannot represent: ingredient **synergy** and **conflict**.

The graph is a directed graph whose nodes are concerns, INCIDecoder functions, individual ingredients, canonical *active groups* (e.g., retinoids, BHA, AHA, vitamin C, niacinamide), and products. Its edges encode ingredient→function (`HAS\_FUNCTION`), function→concern (`TARGETS`), product→ingredient with the INCI position weight \(w(i)\) (`CONTAINS`), product→active (`HAS\_ACTIVE`), and product→concern review support (`REVIEW\_SUPPORTS`). On top of these data-derived edges, a curated knowledge layer adds bidirectional `CONFLICTS\_WITH` and `SYNERGIZES\_WITH` edges between active groups, each weighted by a severity \(\sigma_{ab}\) or strength \(\rho_{ab}\) (for example, retinoids conflict with benzoyl peroxide; vitamin C synergizes with vitamin E and ferulic acid).

For a product \(j\), let \(A_j\) be its set of detected active groups. The knowledge graph contributes a multiplicative ranking factor that penalizes co-occurring conflicting actives and rewards synergistic ones:

\[
f_j=\mathrm{clip}\!\left(\prod_{(a,b)\in \mathcal{C}_j}\bigl(1-0.12\,\sigma_{ab}\bigr)\ \prod_{(a,b)\in \mathcal{S}_j}\bigl(1+0.08\,\rho_{ab}\bigr),\ 0.70,\ 1.25\right),
\]

where \(\mathcal{C}_j\) and \(\mathcal{S}_j\) are the conflicting and synergistic active pairs both present in \(A_j\). Products with fewer than two actives receive \(f_j=1\) (no interaction to assess), and the clip bounds prevent any single product from being over-rewarded or eliminated by this term alone.

The final adaptive ranking score combines the temporally adjusted similarity, the history modifier, and the knowledge-graph factor:

\[
S_j=\text{sim}(u',v_j)\,\cdot\,m_j\,\cdot\,f_j,
\]

where \(u'\) is the temporally adjusted user vector. This makes the recommender *conflict-aware*: a product whose evidence profile aligns well with the user can still be demoted if it pairs incompatible actives, while a well-formulated synergistic product is gently promoted.

#### 2) Routine Construction

For multi-product routine construction, Ruvisa optimizes over routine steps such as cleanser, toner, serum, moisturizer, and SPF. The objective is to maximize concern coverage while penalizing ingredient conflicts. Candidate products are shortlisted per routine step, and combinations are scored by their ability to cover the user's main concerns under budget and compatibility constraints. The same active-ingredient conflict knowledge used in the knowledge-graph factor—encoding incompatibilities such as retinoids with benzoyl peroxide, retinoids with exfoliating acids, vitamin C with benzoyl peroxide, and excessive acid combinations—is applied here at the *routine* level, preventing the generator from selecting individually relevant products that may be unsuitable when combined.

### E. Agentic Recommendation Interface

The final layer of Ruvisa is a tool-grounded conversational agent. The agent allows users to ask natural-language questions such as "What product is recommended for my skin now?", "How has my skin changed over the past three months?", "Why was this product recommended?", or "Can you build a routine for my acne and redness?"

The agent is implemented as a ReAct-style workflow using LangGraph and a locally hosted Ollama LLM. The LLM is not allowed to freely invent rankings or directly query the database. Instead, it receives injected context about the current user profile, latest skin analysis, purchase history, and progress summaries, then decides whether to call deterministic tools. These tools include user-profile retrieval, product search, product-detail retrieval, adaptive product recommendation, routine recommendation, skin-analysis comparison, purchase tracking, outcome evaluation, and knowledge-graph–based product explanation.

This separation between language generation and deterministic tools is central to Ruvisa. Database access, product filtering, adaptive scoring, routine optimization, and knowledge-graph reasoning remain implemented in backend functions; the LLM reasons over tool outputs and converts them into user-facing explanations. This reduces hallucination risk because recommendations are grounded in live catalog state, user history, and computed ranking scores rather than in the model's parametric memory.

The knowledge graph is also what makes the agent's explanations faithful rather than generated. When a user asks why a product was recommended, the explanation tool walks the graph along product→ingredient→function→concern paths for the user's most severe concerns and appends any synergy or conflict notes for the product's actives, so the rationale is traced from concrete ingredient evidence rather than paraphrased by the model.

The agent follows an iterative loop: it receives the user query, reads the injected context, determines whether more information is needed, calls one or more tools, observes the returned evidence, and generates a final response. The API response also records the tools used during the turn, enabling debugging and future evaluation of tool-use correctness.

Through this design, Ruvisa provides more than a ranked product list. It becomes an evidence-grounded decision-support interface that can explain why a product matches the user's current concerns, why another product may be less suitable, how previous outcomes affect current recommendations, and how multiple products can be combined into a safer routine.
