# Ranking & Matching — Expanded Materials & Method

Expanded version of the ranking and adaptive matching subsection, with vector construction details, penalty/boost thresholds, worked examples, routine optimization, and suggested figures.

---

## 1. Problem Setup

Let $\mathcal{C} = \{1, \ldots, C\}$ index skin concerns (e.g., acne, pigmentation, pores). A **user state** is a vector $\mathbf{u} = (u_1, \ldots, u_C)^\top$, where $u_c \geq 0$ reflects current severity or priority for concern $c$ from facial analysis. Each candidate product $j$ is represented by $\mathbf{v}_j = (v_{j,1}, \ldots, v_{j,C})^\top$, where $v_{j,c}$ aggregates ingredient–concern evidence (optionally concentration-weighted) and, when available, review-derived effectiveness for that concern. User and product thus live in a **shared concern space**; matching means identifying products whose $\mathbf{v}_j$ aligns with $\mathbf{u}$.

### 1.1 User vector construction

The user concern vector is built from the facial analysis pipeline. Each detected skin lesion is mapped to one of the 7 concern dimensions, and the score for each concern combines **detection count** (capped at 10 for normalization) with **mean severity**:

$$u_c = \min\!\left(1.0,\; \frac{\text{count}_c}{10}\right) \times \left(0.5 + 0.5 \times \overline{\text{severity}}_c\right)$$

where $\text{count}_c$ is the number of detections mapped to concern $c$, and $\overline{\text{severity}}_c$ is the mean severity score of those detections. This produces values in $[0, 1]$ where higher values indicate more severe or prevalent concern.

**Detection class mapping:**

| Detection class | Mapped concern |
| --- | --- |
| acne, papule, pustule, nodule, cyst | acne |
| blackhead, whitehead, comedone | comedonal_acne |
| dark_spot, pigmentation | pigmentation |
| acne_scars, scar | acne_scars_texture |
| redness | redness |

Wrinkles are handled separately via segmentation: the wrinkle pixel coverage percentage is linearly scaled, with 5% coverage mapping to the maximum concern value of 1.0:

$$u_{\text{wrinkles}} = \min\!\left(1.0,\; \frac{\text{wrinkle\_pct}}{5.0}\right)$$

The pores dimension is set from a dedicated pore analysis score when available.

### 1.2 Product vector construction

Each product's 7-dimensional vector blends **INCI ingredient evidence** with **review effectiveness** (when available):

$$v_{j,c} = \begin{cases} 0.5 \cdot \text{sc}^{\text{ing}}_c + 0.5 \cdot \text{sc}^{\text{rev}}_c & \text{if reviews mention concern } c \\ \text{sc}^{\text{ing}}_c & \text{otherwise} \end{cases}$$

where:

- $\text{sc}^{\text{ing}}_c \in [0, 1]$ is the INCI position-weighted evidence score (maximum position weight among matched ingredients for concern $c$)
- $\text{sc}^{\text{rev}}_c = \frac{\text{eff}_c + 1}{2} \in [0, 1]$ is the review effectiveness remapped from $[-1, 1]$ to $[0, 1]$

The equal weighting (0.5 / 0.5) treats ingredient composition and real-world user outcomes as complementary signals. When no reviews mention a particular concern, the product vector falls back to ingredient evidence alone.

---

## 2. Base Similarity

We measure alignment with **cosine similarity**:

$$\text{sim}(\mathbf{u}, \mathbf{v}_j) = \frac{\mathbf{u}^\top \mathbf{v}_j}{\|\mathbf{u}\| \|\mathbf{v}_j\|} = \frac{\sum_{c \in \mathcal{C}} u_c \, v_{j,c}}{\sqrt{\sum_c u_c^2} \sqrt{\sum_c v_{j,c}^2}}$$

Because $u_c, v_{j,c} \geq 0$, we have $\text{sim} \in [0, 1]$. Cosine similarity compares **direction** in concern space (relative emphasis across concerns), which is convenient when absolute scales of user analysis and product scores differ.

Products with a zero evidence vector (all $v_{j,c} = 0$) or that exceed the user's budget are excluded from ranking.

---

## 3. Adaptive Matching: Outcomes and Temporal Updates

Static similarity ignores whether past recommendations helped. We therefore apply a **modifier** $m_j$ so that the final score is:

$$s_j = s_j^{(0)} \cdot m_j$$

where $s_j^{(0)} = \text{sim}(\mathbf{u}', \mathbf{v}_j)$ is the base cosine similarity (computed on the boosted user vector $\mathbf{u}'$, see Section 3.3), and $m_j \in (0, \infty)$ is the adaptive modifier. The modifier can **penalize** ($m_j < 1$) products associated with poor outcomes, or **boost** ($m_j > 1$) products associated with favorable outcomes.

### 3.1 Outcome attribution

Between two facial analysis scans, the system attributes outcomes to products the user reported using. For each concern $c$, the change in severity $\Delta_c$ is computed between scans. A product is classified based on the balance of worsened vs. improved concerns (using a threshold of $|\Delta_c| > 0.03$ to filter noise):

| Condition | Attributed outcome |
| --- | --- |
| More concerns worsened than improved | **worsened** |
| More concerns improved than worsened | **improved** |
| No concerns changed beyond threshold | **no_change** |
| Equal worsened and improved counts | **mixed** |

### 3.2 Direct outcome modifiers

Products with attributed outcomes receive direct modifiers:

**Table 1: Direct penalty and boost modifiers**

| Attributed outcome | Modifier $m_j$ | Effect |
| --- | ---: | --- |
| worsened | 0.05 | Near-exclusion (95% score reduction) |
| mixed | 0.30 | Strong penalty |
| no_change | 0.40 | Moderate penalty (ineffective product) |
| improved | 2.00 | Strong boost (repurchase signal) |

For **improved** products, additional floors ensure they always surface: if cosine similarity is zero or negative, a floor of $\text{sim} = 0.35$ is applied; after all modifiers, $\text{sim} \geq 0.25$ is enforced.

### 3.3 Ingredient-overlap penalty (Jaccard similarity)

Even for products the user has **never tried**, if a product's active ingredient profile is similar to a previously failed formulation, it is penalized. This encodes the hypothesis that similar active ingredients may produce similar adverse or null responses.

The overlap is measured using **Jaccard similarity** between the evidence-matched ingredient sets:

$$J(I_j, I_{\text{fail}}) = \frac{|I_j \cap I_{\text{fail}}|}{|I_j \cup I_{\text{fail}}|}$$

where $I_j$ is the set of evidence-matched active ingredients for product $j$, and $I_{\text{fail}}$ is the ingredient set of a previously failed product.

**Table 2: Jaccard overlap penalty thresholds**

| Jaccard overlap | Modifier cap | Interpretation |
| ---: | ---: | --- |
| $> 0.6$ | $\leq 0.20$ | High similarity to failed formulation — near-exclusion |
| $> 0.4$ | $\leq 0.50$ | Moderate similarity — strong penalty |
| $> 0.2$ | $\leq 0.75$ | Low similarity — mild penalty |
| $\leq 0.2$ | 1.00 | No penalty |

Symmetrically, products with high ingredient overlap to a previously **improved** formulation receive a boost:

**Table 3: Jaccard overlap boost thresholds**

| Jaccard overlap with improved product | Modifier floor | Interpretation |
| ---: | ---: | --- |
| $> 0.6$ | $\geq 1.40$ | High similarity to successful formulation — strong boost |
| $> 0.4$ | $\geq 1.25$ | Moderate similarity — moderate boost |
| $> 0.25$ | $\geq 1.10$ | Low similarity — mild boost |
| $\leq 0.25$ | 1.00 | No boost |

When both penalties and boosts apply, the direct outcome modifier takes precedence: Jaccard boosts are only applied when $m_j \geq 1.0$ (no existing penalty), and Jaccard penalties are only applied when $m_j = 1.0$ (no direct outcome modifier).

### 3.4 Concern boosting (temporal personalization)

Between scans, if concern $c$ has worsened (severity increase $\Delta_c > 0.03$), the user vector is boosted for that dimension before computing similarity:

$$u'_c = \min\!\left(1.0,\; u_c \times (1 + \Delta_c \times 3)\right)$$

This shifts the matching toward products whose $\mathbf{v}_j$ loads strongly on the worsening dimensions, providing temporal personalization without hand-crafted reordering rules. The cap at 1.0 prevents any single concern from dominating the vector.

**Example:** if acne severity increased by $\Delta = 0.15$ between scans, the acne dimension of the user vector is multiplied by $1 + 0.15 \times 3 = 1.45$ (capped at 1.0 if the boosted value exceeds it).

### 3.5 Summary of adaptive score computation

The complete adaptive scoring for a single product is:

$$s_j = \underbrace{\text{sim}(\mathbf{u}', \mathbf{v}_j)}_{\text{base similarity}} \times \underbrace{m_j}_{\text{modifier}}$$

where:

1. $\mathbf{u}'$ is the user vector after concern boosting (Section 3.4)
2. The base similarity is cosine similarity between $\mathbf{u}'$ and the product vector $\mathbf{v}_j$
3. $m_j$ is determined by (in priority order):
   - Direct outcome modifier if product was previously used (Table 1)
   - Jaccard penalty if ingredient overlap with a failed product exceeds 0.2 (Table 2)
   - Jaccard boost if ingredient overlap with an improved product exceeds 0.25 (Table 3)
   - Default: $m_j = 1.0$

Products are ranked by $s_j$ in descending order. Secondary tie-breakers include declared skin-type compatibility (products matching the user's skin type are prioritized) and price (lower price preferred among ties).

---

## 4. Relation to Multi-Product Routine Matching

End-to-end routine construction adds a **combinatorial optimization layer** on top of single-product ranking. The system selects one product per routine step (Cleanser → Toner → Serum → Moisturizer → SPF) to maximize concern coverage while penalizing ingredient conflicts.

### 4.1 Objective function

The routine optimization objective is:

$$\text{score} = \underbrace{\sum_{c=1}^{C} u_c \cdot \max_{j \in R} v_{j,c}}_{\text{coverage}} - \lambda \cdot \underbrace{\sum_{(i,k) \in \binom{R}{2}} \text{conflict}(i, k)}_{\text{conflict penalty}}$$

subject to: one product per step, total cost $\leq$ budget.

**Coverage** uses the **maximum** product score per concern across the routine, weighted by the user's concern priorities. This avoids double-counting redundant coverage (e.g., two acne products don't help more than the better one).

**Conflict penalty** sums pairwise ingredient conflict scores between all products in the routine. The trade-off parameter $\lambda$ controls how aggressively conflicts are penalized ($\lambda = 2.0$ default, $\lambda = 5.0$ in the API for stricter safety).

### 4.2 Ingredient conflict pairs

Expert-derived conflict rules encode known active ingredient incompatibilities:

**Table 4: Ingredient conflict groups and severity**

| Group A | Group B | Severity | Rationale |
| --- | --- | ---: | --- |
| Retinol, Retinyl, Adapalene, Tretinoin | Benzoyl Peroxide | 1.0 | Retinoids are deactivated by benzoyl peroxide |
| Retinol, Retinyl, Adapalene, Tretinoin | Glycolic, Lactic Acid, Mandelic, PHA | 0.9 | Combined exfoliation risk — excessive irritation |
| Retinol, Retinyl, Adapalene, Tretinoin | Salicylic Acid | 0.9 | Retinoid + BHA over-exfoliation |
| Ascorbic Acid (Vitamin C) | Benzoyl Peroxide | 1.0 | Vitamin C is oxidized and deactivated |
| Ascorbic Acid (Vitamin C) | Niacinamide | 0.5 | Potential flushing at high concentrations |
| AHA (Glycolic, Lactic, Mandelic) | BHA (Salicylic) | 0.7 | Dual acid over-exfoliation |
| AHA (Glycolic, Lactic, Mandelic) | AHA (same group) | 0.5 | Redundant/excessive exfoliation |
| Copper Peptide (GHK-Cu) | Ascorbic Acid (Vitamin C) | 0.8 | Copper ions destabilize Vitamin C |

The total pairwise conflict score is capped at 1.0 per product pair.

### 4.3 Candidate shortlisting

For computational tractability, the system shortlists the **top 8 products per routine step** (ranked by cosine similarity to the user vector, with a $\times 1.2$ bonus for skin-type compatibility). The exhaustive search then evaluates all combinations of shortlisted candidates ($\leq 8^5 = 32{,}768$ combinations in the worst case). Products with direct outcome penalties from adaptive matching are excluded from routine candidates.

---

## 5. Suggested Figures

### Figure A — Adaptive matching score computation

A flowchart showing the complete adaptive scoring pipeline for a single product:

1. **Inputs:** User vector $\mathbf{u}$ (from facial analysis) + product vector $\mathbf{v}_j$ (ingredient + review)
2. **Concern boosting:** Worsened concerns increase $\mathbf{u}$ → $\mathbf{u}'$
3. **Cosine similarity:** $\text{sim}(\mathbf{u}', \mathbf{v}_j)$
4. **Modifier computation:** Direct outcome → Jaccard overlap → default 1.0
5. **Final score:** $s_j = \text{sim} \times m_j$
6. **Ranking:** Sort by $s_j$ descending

Show branching paths for the modifier: penalty branch (failed products, Jaccard penalty) and boost branch (improved products, Jaccard boost, repurchase).

**Caption:** Adaptive matching score computation. The base cosine similarity (computed on the boosted user vector) is scaled by a history-aware modifier that penalizes products associated with poor outcomes and boosts products with favorable outcomes. Ingredient-overlap (Jaccard) extends penalties and boosts to unseen products with similar formulations.

### Figure B — Concern boosting worked example

A visual showing two user vectors side-by-side:

- **Before boosting:** $\mathbf{u} = [0.6, 0.3, 0.2, 0.0, 0.4, 0.1, 0.1]$
- **After boosting** (acne worsened by $\Delta = 0.15$, redness worsened by $\Delta = 0.10$):
  - acne: $0.6 \times 1.45 = 0.87$
  - redness: $0.1 \times 1.30 = 0.13$
  - Others unchanged
- **Result:** $\mathbf{u}' = [0.87, 0.3, 0.2, 0.0, 0.4, 0.13, 0.1]$

Show as a radar chart or bar chart with before/after overlay to visualize the directional shift.

**Caption:** Concern boosting shifts the user vector toward worsening concerns, causing cosine similarity to favor products that specifically address those dimensions.

### Figure C — Jaccard ingredient overlap

A Venn diagram example showing:

- **Failed product ingredients:** {Niacinamide, Salicylic Acid, Zinc PCA, Glycerin, Hyaluronic Acid}
- **Candidate product ingredients:** {Niacinamide, Salicylic Acid, Zinc PCA, Tea Tree Oil, Allantoin}
- **Intersection:** 3 ingredients, **Union:** 7 ingredients
- **Jaccard:** 3/7 = 0.43 → modifier capped at 0.50

**Caption:** Jaccard similarity between a failed product's active ingredients and a candidate product. High overlap (> 0.4) triggers a penalty modifier, deprioritizing formulations similar to those that produced poor outcomes.

### Figure D — Routine optimization objective

A diagram showing:

- **Left:** 5 routine steps (Cleanser, Toner, Serum, Moisturizer, SPF) with candidate pools
- **Center:** Coverage computation — for each of 7 concerns, take the max score across the 5 selected products, weighted by user priority
- **Right:** Conflict penalty — pairwise checks between all selected products
- **Bottom:** Final score = coverage − $\lambda$ × conflict

**Caption:** Routine optimization selects one product per step to maximize user-weighted concern coverage while penalizing pairwise ingredient conflicts. The trade-off parameter $\lambda$ controls conflict aversion.
