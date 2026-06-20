# Ruvisa — IEEE Conference Paper: Introduction (Draft)

> **Formatting note for the 8-page limit.** In IEEE conference style the *Introduction* is a single tight section (~1 two-column page) that motivates the problem, exposes the gap in one or two sentences per prior approach, states the proposed system, and ends with an explicit, numbered list of contributions. The detailed six-part survey you wrote (vision, INCI text, reviews, recommenders, LLM agents) should be **moved into a separate `II. Related Work` section** rather than living inside the Introduction. The version below assumes that split.

---

## I. Introduction

Looking better is a near-universal aspiration, and the global beauty market is projected to reach USD 889.33 billion by 2027 [8]. Yet skincare shopping has not kept pace: online retail runs on an exhausting loop of keyword search, marketing claims, influencer trends, and unreliable reviews, while clinically grounded guidance stays in costly, inaccessible offline clinics. The result is measurable waste—nearly 49% of users keep unopened products for over a year [9], and about 70% feel overwhelmed by choice [10]. Shoppers are not short on products; they lack trustworthy, personalized reasoning about which product fits their skin, budget, and history.

The signals a recommendation needs are scattered across incompatible modalities: the skin's state lives in a *photograph*; a product's effect is encoded in its *INCI list*, where ingredient order carries meaning [1], [7]; lived effectiveness is buried in noisy, sparse *reviews* [5]; and whether the last recommendation worked depends on *purchase history and follow-up scans* most systems never revisit. Existing platforms optimize one signal in isolation and for engagement, so personalization stays shallow and recommendations remain hard to justify.

Three gaps follow. **(i) Modality silos:** vision outputs rarely align with the concern vocabulary used for product matching [2], [3]. **(ii) Ungrounded dialogue:** LLMs give fluent advice but are weakly bound to SKU-level evidence and real user state, inviting hallucination [6]. **(iii) Static personalization:** recommenders fit preference once and never re-rank on repeat scans or post-purchase outcomes [4]. Across all three runs a tension between explainability and flexibility.

We present **Ruvisa**, an end-to-end skincare e-commerce system that closes these gaps in one deployable architecture. It turns a selfie into structured concern scores via computer vision with face-parsing localization, derives product intelligence from INCI lists using position-aware evidence and a DeBERTa multi-label model, and aggregates reviews into per-concern signals. Every signal is projected onto a shared seven-concern schema and matched to the user by cosine similarity under adaptive, feedback-driven re-ranking. A skincare knowledge graph adds ingredient synergy/conflict for conflict-aware ranking and human-readable justifications, and a ReAct-style agent grounds conversation in live user and catalog state [6].

Our contributions are: (1) a schema-aligned multimodal pipeline unifying image, ingredient, and review signals onto one shared seven-concern representation; (2) a hybrid, explainable knowledge-graph–aware recommender with adaptive post-purchase feedback; (3) a grounded agentic interface that binds a ReAct-style LLM agent to live user history and catalog evidence; and (4) a documented, reproducible deployment under real storage and latency constraints. The remainder of this paper is organized as follows. Section II reviews related work; Section III details the Ruvisa architecture; Section IV describes the deployment; Section V presents the evaluation; and Section VI concludes.

---

## II. Related Work

Ruvisa draws on four threads—consumer skin analysis, ingredient/review product modeling, recommendation, and agentic LLMs—reviewed below.

**Skincare commerce and computer-vision skin analysis.** Online skincare retail has moved from static catalogs to interactive self-assessment tools that estimate lesions, pigmentation, and aging from facial images [1], [2], with ingredient-transparency tools reflecting demand for claim-independent guidance [16]. Vision research spans lesion detection, severity grading, and segmentation [2], [3]; recent systems jointly detect and grade acne at ~90% accuracy across capture conditions [13], and efficient backbones such as BiSeNet enable real-time face parsing for localization [12]. Yet robustness to lighting, pose, and domain shift remains hard [2], [3], [13], and most work stops at diagnosis without aligning outputs to a downstream recommendation schema—exactly the integration Ruvisa targets.

**Cosmetic ingredients (INCI) and reviews.** INCI lists give an ordered, formulation-grounded product representation, and ingredient-based modeling supports effect prediction and recommendation [1], [7]; because order encodes approximate concentration, sequence-aware analysis aids cosmetic reasoning [1], and ingredient-similarity engines surface brand-independent alternatives [16]. Pretrained encoders such as DeBERTa [11] provide strong multi-label text classification, which we use offline to validate ingredient-derived labels. Reviews add experiential signal but are noisy and sparse; aspect-based sentiment analysis targets fine-grained opinions yet struggles with implicit, domain-specific mentions [5], [14]. Ruvisa converts INCI into a position-aware per-concern vector and aggregates reviews via a transparent rule-based scheme onto the same schema.

**Recommendation and knowledge graphs.** Recommenders are typically collaborative, content-based, or hybrid, with hybrids mitigating cold-start and sparsity [4]. Skincare recommendation is inherently hybrid—skin state, ingredients, review outcomes, and purchase history are complementary—and recent two-tower models fuse user and ingredient embeddings for biologically informed suggestions [16]. Knowledge-graph recommenders inject relational side information and make reasoning explainable through path-level explanations [15], [17]. Ruvisa combines concern-space similarity with a knowledge graph encoding ingredient synergy/conflict for conflict-aware, interpretable re-ranking.

**LLMs, agentic interfaces, and agentic commerce.** ReAct-style agents interleave reasoning and tool use to ground LLMs in external systems [6], underpinning the emerging paradigm of *agentic commerce*, where agents research, compare, and transact on a user's behalf via open checkout/payment protocols [18]. Such efforts emphasize transactional plumbing and generic search, rarely coupling the agent to domain-specific, evidence-grounded reasoning about individual fit. Ruvisa fills this gap: its agent is bound to live user state, SKU-level evidence, knowledge-graph explanations, and a self-correcting recommender, making it an evidence-grounded instance of agentic commerce with autonomous checkout as a planned extension. Overall, prior work advances single components in isolation; Ruvisa's contribution is integrative—aligning these modalities on one shared schema within a single deployed system.

---

## References (additions for Related Work)

> Existing references [1]–[10] are unchanged. The entries below are the new sources cited in Section II; please verify author lists/page numbers against the official versions before submission.

- **[11]** P. He, X. Liu, J. Gao, and W. Chen, "DeBERTa: Decoding-Enhanced BERT with Disentangled Attention," in *Proc. Int. Conf. Learning Representations (ICLR)*, 2021.
- **[12]** C. Yu, J. Wang, C. Peng, C. Gao, G. Yu, and N. Sang, "BiSeNet: Bilateral Segmentation Network for Real-Time Semantic Segmentation," in *Proc. European Conf. Computer Vision (ECCV)*, 2018, pp. 325–341.
- **[13]** Z. Zhang *et al.*, "Evaluation of an Acne Lesion Detection and Severity Grading Model for Chinese Population in Online and Offline Healthcare Scenarios," *Scientific Reports*, vol. 15, 2024, Art. no. 84670. (AcneDGNet)
- **[14]** Y. Chen *et al.*, "A Systematic Review of Aspect-Based Sentiment Analysis: Domains, Methods, and Trends," *Artificial Intelligence Review*, vol. 57, 2024.
- **[15]** Q. Guo *et al.*, "A Survey on Knowledge Graph-Based Recommender Systems," *IEEE Trans. Knowledge and Data Engineering*, vol. 34, no. 8, pp. 3549–3568, 2022.
- **[16]** A. Author *et al.*, "A Hybrid Deep Neural Architecture for Personalized Skincare Recommendation" (Multi-Head Attention Two-Tower model on Amazon Beauty), *Science World Journal*, 2024.
- **[17]** S. Author *et al.*, "Review of Explainable Graph-Based Recommender Systems," *ACM Computing Surveys*, 2024.
- **[18]** OpenAI and Stripe, "Agentic Commerce Protocol (ACP): An Open Standard for AI-Agent-Initiated Commerce," 2025. [Online]. Available: https://www.agenticcommerce.dev. (Alternatively cite McKinsey & Co., "The Agentic Commerce Opportunity," 2025, for the conceptual framing.)
