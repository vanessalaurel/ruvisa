# Ruvisa SEEDfund Pitch Deck

**Format:** 13-slide investor / startup pitch deck  
**Audience:** SEEDfund, early-stage innovation grant, startup demo panel  
**Positioning:** AI-powered skincare commerce for Hong Kong and Southeast Asia  

---

## Slide 1 — Title

### Ruvisa
**AI skincare commerce that turns selfies into science-backed product decisions.**

Ruvisa replaces marketing-led beauty shopping with an adaptive AI beauty agent that analyzes skin, understands product ingredients, learns from real outcomes, and recommends what each user should actually buy next.

**Tagline options**
- Your skin. Your evidence. Your next best product.
- Beauty shopping, guided by AI and ingredient science.
- From selfie to smarter skincare.

**Speaker note:**  
Ruvisa is not just a skincare analyzer and not just an online shop. It is an AI-powered commerce layer that connects skin analysis, ingredient intelligence, user reviews, purchase history, and a conversational beauty agent into one personalized shopping journey.

---

## Slide 2 — Problem

### Skincare shopping is overloaded, expensive, and driven by claims.

Consumers face thousands of products, influencer trends, promotional claims, and conflicting reviews, but they still lack a trusted answer to one simple question: **"What will work for my skin now?"**

**Pain points**
- Product claims are often easier to understand than ingredient evidence.
- Reviews are noisy and difficult to compare across skin types and concerns.
- Users buy products that do not match their current skin condition.
- Skincare waste grows when products are opened, tried briefly, or left unused.
- Existing e-commerce platforms recommend based on popularity, brand, or sales logic rather than measurable skin progress.

**Hong Kong evidence:** Hong Kong consumers spend an average of about **HK$3,100 per year** on skincare, while nearly **49%** reported leaving skincare products unused for over a year, according to Hong Kong Consumer Council reporting and coverage by SCMP and the Consumer Council.

**Speaker note:**  
The problem is not that consumers do not care about skincare. The problem is that they care so much that they over-search, over-buy, and still cannot verify which products are suitable for their real skin condition.

---

## Slide 3 — Market Opportunity

### Beauty commerce is large, growing, and increasingly digital across Asia.

Ruvisa starts with Hong Kong as a high-spending, beauty-aware launch market, then expands into Southeast Asia where beauty and personal care e-commerce is already mainstream.

**Market signals**
- Hong Kong skincare market reports estimate growth from around **USD 3.1B in 2025** to **USD 4.0B by 2030**.
- Southeast Asia beauty and personal care revenue is projected around **USD 36.14B in 2025**, with online sales contributing a large share of market revenue.
- Personalized skincare is a fast-growing category globally, supported by AI diagnostics, digital skin tests, and direct-to-consumer commerce.

**Why now**
- Consumers are comfortable with selfie-based assessment.
- Beauty shopping is increasingly online, social, and mobile-first.
- Retailers need higher conversion and lower return/frustration loops.
- AI agents can now explain, compare, and personalize in natural language.

**Speaker note:**  
The timing is strong because the market already spends heavily on skincare, but discovery is still broken. Ruvisa sits at the intersection of beauty commerce, AI personalization, and evidence-based product matching.

---

## Slide 4 — Solution

### Ruvisa is a personal AI beauty agent for skincare shopping.

Ruvisa helps users move from uncertainty to action:

1. **Upload a selfie**  
   The system detects skin concerns and converts them into a structured concern profile.

2. **Understand the product**  
   Ruvisa reads ingredient evidence, review signals, and product metadata.

3. **Match skin to products**  
   A hybrid recommender compares the user's skin state with product evidence.

4. **Learn over time**  
   The system tracks purchases and future scans to adapt recommendations.

5. **Explain in conversation**  
   A tool-using AI agent answers questions, compares progress, and recommends routines.

**One-line user promise:**  
Ruvisa tells users what to buy, why it matches their skin, and whether it keeps working over time.

**Speaker note:**  
The key difference is continuity. Ruvisa does not stop after one quiz or one scan. It learns from skin progress and product outcomes, which creates an evolving personalization loop.

---

## Slide 5 — Product Demo Flow

### From selfie to recommended routine in one guided journey.

**User journey**
- Register or log in
- Upload a face image
- Receive skin concern analysis and overall score
- View concern-specific product recommendations
- Ask the Ruvisa agent follow-up questions
- Save, purchase, or track products
- Return later for progress comparison and adaptive re-ranking

**Example questions Ruvisa can answer**
- "What product is recommended for my skin now?"
- "How has my skin changed over the past 3 months?"
- "Why is this serum recommended for acne scars?"
- "Can you build me a routine under my budget?"
- "Should I avoid products similar to one that broke me out?"

**Speaker note:**  
For a demo panel, the clearest demo is: upload image, show concern profile, show product ranking, then ask the agent why the top product was recommended.

---

## Slide 6 — Technology

### A multimodal AI system built for explainable skincare commerce.

Ruvisa combines four intelligence layers:

**1. Skin image analysis**
- YOLO-based lesion detection
- ResNet severity classification
- Face parsing and region mapping
- Wrinkle segmentation
- Output: seven-dimensional skin concern vector

**2. Product intelligence**
- INCI ingredient evidence
- Review-based concern sentiment
- DeBERTa multi-label product understanding
- Product vector mapped to the same seven concern dimensions

**3. Adaptive ranking**
- Cosine similarity between user skin vector and product vector
- Skin type compatibility filtering
- Boosts for previously helpful products
- Penalties for poor outcomes and ingredient overlap with failed products

**4. Agentic assistant**
- FastAPI backend
- LangGraph ReAct-style tool loop
- Local Ollama LLM runtime
- Tools for product search, recommendation, routines, purchases, outcomes, and progress comparison

**Speaker note:**  
The technical advantage is that every layer speaks the same concern language. This makes recommendations explainable instead of being a black-box "because people like you bought this" system.

---

## Slide 7 — Differentiation

### Ruvisa is not another quiz, marketplace filter, or generic chatbot.

| Existing approach | Limitation | Ruvisa advantage |
| --- | --- | --- |
| Beauty quizzes | Static self-reported answers | Uses image-derived skin state |
| Product filters | User must already know what to search | Recommends based on concern evidence |
| Influencer/review shopping | Noisy, subjective, trend-driven | Aggregates review signals by concern |
| Generic AI chatbot | Can hallucinate product advice | Uses tools tied to database and ranking logic |
| Standard recommender | Often based on popularity or sales | Uses skin-product vector matching and outcomes |

**Core moat**
- Shared concern representation from face to catalog to ranker
- Ingredient-first product understanding
- Longitudinal learning from purchases and progress
- Agentic interface that can explain and act, not just answer

**Speaker note:**  
Ruvisa's moat is the loop: analyze, recommend, purchase, track, adapt. Each loop creates better personalization and stronger user trust.

---

## Slide 8 — Validation From Prototype

### A working end-to-end prototype has already been built.

**Current implemented capabilities**
- FastAPI backend with authentication, analysis, chat, products, purchases, likes, bag, and recommendations
- React/Vite frontend with user-facing app experience
- SQLite product and user data layer
- Product catalog and review evidence pipeline
- Computer vision skin analyzer
- Adaptive recommendation engine
- Conversational agent with eight domain tools
- Cloudflare Pages + VPS deployment path in progress

**Internal performance and system evidence**
- Skin analyzer converts raw images into a usable seven-dimensional concern vector.
- Severity classifier reached **0.9938 test accuracy** on the saved severity test set.
- INCI-grounded DeBERTa product understanding outperformed claims-oriented inputs in logged cross-validation summaries.
- Adaptive ranking changed top product order when user history provided strong evidence, while staying sparse and interpretable.
- Example ranking snapshot scored **911 products**, with **817 skin-type compatible products** and selective boosted results.

**Speaker note:**  
This is not an idea-only project. The current work already demonstrates the full technical pipeline, even though external user validation and production hardening are the next priorities.

---

## Slide 9 — Business Model

### Ruvisa can monetize through commerce, partnerships, and premium intelligence.

**Revenue streams**
- **Affiliate / commission model:** earn commission from product purchases routed through Ruvisa.
- **Brand / retailer partnerships:** provide explainable recommendation placement without hiding evidence logic.
- **Premium consumer plan:** advanced progress tracking, routine optimization, and ingredient conflict warnings.
- **B2B skin intelligence API:** offer product matching, review intelligence, or skin concern profiling to beauty retailers.

**Why brands and retailers care**
- Better conversion from personalized recommendations
- Better trust than generic sponsored placement
- More repeat engagement through progress tracking
- Richer consumer insight across concerns, products, and outcomes

**Speaker note:**  
Ruvisa can begin with affiliate commerce because it is the simplest to test, then grow into retailer partnerships and B2B intelligence once user and recommendation data validate demand.

---

## Slide 10 — Go-To-Market

### Start narrow in Hong Kong, then expand through SEA beauty channels.

**Phase 1: Hong Kong beta**
- Target skincare-aware students, young professionals, and beauty shoppers aged 18-35.
- Launch with a simple web app and mobile-friendly scan flow.
- Use Sephora HK-style catalog coverage as the first recognizable product universe.
- Recruit beta users through universities, beauty communities, TikTok/Instagram, and skincare forums.

**Phase 2: Trust and retention**
- Weekly or monthly skin progress check-ins
- "Before vs now" skin journey reports
- Personalized routine reminders
- Product outcome tracking after purchase

**Phase 3: Southeast Asia expansion**
- Expand product catalog coverage to major marketplaces and retailers.
- Localize for Singapore, Malaysia, Indonesia, Thailand, and the Philippines.
- Partner with beauty creators, clinics, and indie skincare brands.

**Speaker note:**  
The first growth goal is not to list every product in Asia. It is to prove that users trust Ruvisa recommendations more than browsing and that they return to track progress.

---

## Slide 11 — Roadmap

### From prototype to investable beauty commerce platform.

**Next 3 months**
- Stabilize deployment with domain, HTTPS, monitoring, and database backups
- Improve onboarding and demo flow
- Add analytics for scans, clicks, recommendations, and purchases
- Run beta test with 50-100 users

**Next 6 months**
- Improve skin analyzer robustness across lighting, pose, and skin tones
- Expand product catalog and review evidence coverage
- Add recommendation explanations and ingredient conflict warnings in UI
- Build affiliate purchase tracking and conversion dashboard

**Next 12 months**
- Launch merchant/brand partnership pilot
- Add makeup and skin-tone recommendation module
- Add fulfillment-aware recommendations
- Build consumer mobile app or PWA
- Prepare for regional rollout across Southeast Asia

**Speaker note:**  
The roadmap is designed to move from a strong technical prototype into a measurable commercial product with real users, retention, and purchase conversion.

---

## Slide 12 — Team

### Founder-led product with AI, software, and beauty-market insight.

**Founder / Project Lead: Vanessa Laurel**

**Current strengths**
- Built the working Ruvisa prototype across frontend, backend, computer vision, product intelligence, recommendation logic, and agentic workflow.
- Developed the thesis-grade research foundation behind the product, including skin analysis, INCI evidence, review analysis, DeBERTa product labeling, and adaptive ranking.
- Understands the target user pain point from the consumer side: beauty shoppers want personalization, but most e-commerce guidance is still driven by claims, trends, and generic filtering.

**Near-term hiring / advisor needs**
- Dermatology / cosmetic science advisor for validation and safety boundaries
- Beauty retail or e-commerce advisor for partnerships
- AI / MLOps advisor for production model reliability
- Growth / creator marketing support for Hong Kong beta launch

**Speaker note:**  
This slide should be personalized with your photo, university/program, technical role, and any relevant awards, thesis achievements, hackathons, startup experience, or beauty/e-commerce exposure.

---

## Slide 13 — Funding Ask

### SEEDfund will help Ruvisa move from working prototype to market pilot.

**Funding use**
- Product hardening: deployment, security, monitoring, and data backup
- Model improvement: better skin-image datasets and testing across more skin tones
- Catalog expansion: more products, reviews, and ingredient evidence
- User validation: beta recruitment, usability testing, and feedback loops
- Commercial testing: affiliate links, partner outreach, and conversion analytics

**Milestones with support**
- Launch stable public beta
- Validate 50-100 early users
- Track scan-to-recommendation-to-click behavior
- Demonstrate repeat skin progress tracking
- Prepare first brand or retailer partnership pilot

**Closing statement**
Ruvisa is building the intelligence layer between beauty consumers and skincare products, helping users buy with evidence instead of hype.

**Speaker note:**  
The key ask is not just money for development. It is support to validate whether this AI-commerce model can become a trusted skincare decision platform for Hong Kong and Southeast Asia.

---

# Appendix — Source Links And Evidence Notes

## Market Sources

- Hong Kong Consumer Council, skincare product survey and wastage discussion: <https://www.consumer.org.hk/en/press-release/p-542-skincare-product-user-opinion-survey>
- SCMP coverage of Hong Kong skincare spending and unused product waste: <https://www.scmp.com/news/hong-kong/hong-kong-economy/article/3159789/hongkongers-spend-average-hk3100-skincare-products>
- Hong Kong skincare market projection summary via StrategyHelix: <https://strategyh.com/report/skin-care-products-market-in-hong-kong/>
- Southeast Asia beauty and personal care market summary via Statista: <https://www.statista.com/outlook/cmo/beauty-personal-care/southeast-asia>
- TMO Group Southeast Asia skincare e-commerce market insights: <https://www.tmogroup.asia/insights/sea-skincare-ecommerce-market/>
- Personalized skincare market report via Grand View Research: <https://grandviewresearch.com/industry-analysis/personalized-skin-care-products-market-report>

## Ruvisa Internal Evidence

- Product and technical summary: `README.md`
- Backend routes: `api/main.py`, `api/routes.py`
- Agent tools and ranking logic: `agent/tools.py`, `agent/graph.py`, `agent/prompts.py`
- Frontend user experience: `frontend/src/App.jsx`, `frontend/src/api.js`
- Skin analyzer results: `docs/results_discussion_skin_analyzer_revised.md`
- Ranking results: `docs/results_discussion_ranking_matching_revised.md`
- Conclusion and future work: `docs/conclusion_ruvisa_revised.md`
- Agentic system explanation: `docs/agentic_system_cited.md`

