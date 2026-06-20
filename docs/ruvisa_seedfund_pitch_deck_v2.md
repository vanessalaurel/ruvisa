# Ruvisa SEED Fund Pitch Deck

**Deck length:** 12 slides  
**Purpose:** Hong Kong SEED fund / early-stage startup application  
**Positioning:** AI skincare commerce and beauty decision platform for Hong Kong and Southeast Asia  
**Funding scenario:** HKD 100,000 pilot support  

---

## Slide 1 — Title

### Ruvisa

**AI skincare commerce that turns selfies into science-backed beauty decisions.**

Ruvisa is a launched AI-powered skincare and beauty platform that analyzes a user's face, understands their skin concerns, matches them with evidence-backed products, and learns from their purchases and skin progress over time.

**Vision:**  
To become the trusted AI beauty agent for skincare, makeup, and aesthetic treatment discovery across Hong Kong and Southeast Asia.

**Suggested visual:**  
Hero mockup: user selfie → skin analysis → product recommendation → AI agent explanation.

**Speaker note:**  
Ruvisa is not only a product recommendation website. It is an AI decision layer for beauty commerce, helping users choose products and eventually clinics based on skin needs, budget, ratings, trend signals, and real outcomes.

---

## Slide 2 — The Problem

### Beauty shoppers are overwhelmed by products, promotions, and treatment choices.

A typical user has a limited budget and a visible concern, for example acne scars, pigmentation, wrinkles, or skin texture. She wants the best product or treatment she can afford, but the market gives her too many options: influencer recommendations, shop seller hype, seasonal discounts, first-trial promotions, beauty clinic packages, and conflicting reviews.

**Hong Kong pain point**

- Hong Kong consumers spend about **HKD 3,100 per year** on facial skincare.
- Nearly **49%** reported keeping unused skincare products unopened for more than a year.
- The Hong Kong Consumer Council warned that buying full product ranges based on brand hype can cause waste and poor product fit.
- Beauty service complaints in Hong Kong also show trust issues around low-cost trial offers, high-pressure sales, prepaid packages, and unclear treatment value.

**Southeast Asia pain point**

- Beauty discovery is increasingly fragmented across Shopee, Lazada, TikTok Shop, Instagram, YouTube, and influencer content.
- Consumers are often overwhelmed with online beauty choices, and brands are already experimenting with AI beauty concierge tools to reduce choice overload.
- TikTok and KOL-driven commerce can accelerate discovery, but it can also make it harder for users to know whether a product fits their actual skin condition.

**Why this matters**

- Users do not know which ingredient, product, routine, or treatment matches their concern.
- Reviews are hard to compare because skin type and concern context are missing.
- Product claims are easier to market than they are to verify.
- Product and clinic promotions are scattered across different platforms.
- E-commerce sites optimize search and sales, not long-term skin outcomes.

**Suggested visual:**  
Messy beauty shelf + clinic promotion cards + quote: "I have a budget. What actually works for my skin?"

**Speaker note:**  
The pain point is not only buying the wrong product. It is the whole decision process: what product to buy, what routine to follow next, whether a treatment is needed, and which promotion or clinic is actually worth trusting.

---

## Slide 3 — Market Opportunity

### Hong Kong is the launch market; Southeast Asia is the expansion market.

Ruvisa targets a growing online beauty market where consumers are already comfortable buying skincare digitally, but still lack trusted personalization.

**Market signals**

- Hong Kong beauty and personal care e-commerce revenue is projected at about **USD 277.4M in 2025**, with growth expected through 2029.
- Hong Kong skincare demand is supported by premium beauty culture, social media influence, and increasing clean/sustainable beauty interest.
- Southeast Asia beauty and personal care is a large regional market, with strong e-commerce activity across Shopee, Lazada, and TikTok Shop.
- Reports on Southeast Asia beauty e-commerce show major beauty GMV on Shopee and rapid TikTok Shop growth, with beauty discovery increasingly influenced by content, live commerce, and KOL sales.
- Personalized skincare and AI skin analysis are fast-growing global categories, driven by AI diagnostics, online consultations, and customized product discovery.

**Why now**

- Consumers accept selfie-based assessment.
- Retailers need higher conversion and repeat engagement.
- AI agents can explain recommendations in natural language.
- Beauty clinics and treatments are also moving toward online discovery and comparison.

**Suggested visual:**  
Map: Hong Kong → Singapore/Malaysia/Indonesia/Thailand/Philippines.

**Speaker note:**  
The market is not only product retail. The larger opportunity is the full beauty decision journey: products, routines, makeup, clinics, and treatments.

---

## Slide 4 — The Solution

### Why Ruvisa: one free AI beauty journey, from analysis to action.

Ruvisa helps users answer the questions beauty shoppers struggle with every day:

- **What is happening to my skin?**
- **What product should I use next?**
- **What routine fits my budget and concern?**
- **Is my skin improving or getting worse?**
- **Do I need a product, makeup, or a professional treatment?**
- **Where can I book the best treatment offer I can afford?**

**What Ruvisa gives users for free**

1. **Free skin analysis** from a selfie, converted into clear skin concern scores.
2. **Personalized product recommendations** based on skin needs, ingredient evidence, reviews, price, and skin type.
3. **Routine recommendation** that suggests what to use next, not just one isolated product.
4. **Beauty journey tracking** so users can compare progress after purchases or treatments.
5. **AI explanation** that tells users why each product or routine is recommended.
6. **Future treatment booking** that connects users to clinics by concern, treatment goal, rating, promotion, quality, and budget.

**Why it is different**

- Ruvisa is not only a beauty shop.
- Ruvisa is not only a skin analyzer.
- Ruvisa is not only a chatbot.
- Ruvisa is a personalized beauty decision assistant that connects analysis, products, routines, progress, promotions, and treatments.

**Core promise**
Users can stop guessing, stop jumping between many shops and clinic pages, and start making beauty decisions based on their actual skin condition, budget, and progress.

**Suggested visual:**  
Journey loop: Free scan → Personalized routine → Product/treatment recommendation → Purchase/book → Track progress → Smarter next recommendation.

**Speaker note:**  
This slide should make the panel feel the user value. Ruvisa gives users a free entry point, then keeps them inside a personalized beauty journey where every recommendation becomes more relevant over time.

---

## Slide 5 — Product Status And User Flow

### The Ruvisa web app is launched and ready for production pilot testing.

Ruvisa already has a working web application with frontend, backend, database, AI analysis, recommendation, and agentic chat components connected.

**Live product flow**

- Register / login
- Upload face image
- Receive skin analysis and concern score
- View personalized product recommendations
- Ask Ruvisa AI for explanations and routines
- Save products, add to bag, like items, and record purchases
- Track skin journey and compare progress over time

**Current production-readiness**

- Deployed frontend through Cloudflare Pages
- Backend deployed on VPS with Nginx/HTTPS setup
- FastAPI backend with `/api` routes and `/health` endpoint
- Product and user data stored through SQLite service layer
- Prepared for beta users and demo evaluation

**Suggested visual:**  
Screenshots from Ruvisa app: home, skin upload, recommendation, AI chat.

**Speaker note:**  
For SEED fund, the key message is that the product is not just theoretical. The foundation is already launched and now needs pilot users, better data, and commercial testing.

---

## Slide 6 — Technology And AI Moat

### Ruvisa combines computer vision, product intelligence, recommender systems, and agentic AI.

**1. Skin Analyzer**

- Detects acne-related lesions and visible skin concerns.
- Estimates severity and maps concerns to face regions.
- Converts image results into a seven-dimensional concern vector.

**2. Product Intelligence**

- Uses INCI ingredient evidence instead of relying only on product claims.
- Aggregates review signals by concern and skin type.
- Uses DeBERTa-style multi-label product understanding for concern prediction.

**3. Adaptive Recommender**

- Matches user skin vectors with product evidence vectors using cosine similarity.
- Adds skin-type compatibility, price, review evidence, and ingredient overlap.
- Learns from purchases and outcomes to boost helpful products and reduce poor matches.

**4. Agentic AI**

- Uses a tool-calling AI agent to retrieve user profile, compare analyses, recommend products, build routines, and explain decisions.
- The agent does not guess blindly; it calls deterministic backend tools.

**Suggested visual:**  
Four-layer architecture stack: Vision → Product Intelligence → Adaptive Ranking → Agentic AI.

**Speaker note:**  
The technical moat is the shared concern space. The user's face and the product catalog are represented in the same concern dimensions, which makes matching explainable and measurable.

---

## Slide 7 — Beyond Products: Beauty Clinic And Treatment Matching

### Ruvisa can grow from skincare commerce into a beauty decision marketplace.

Users do not only need products. Some concerns may require professional treatments such as facials, acne treatments, pigmentation treatments, laser, peeling, extraction, or consultation with clinics. Today, users must compare different clinic websites, Instagram pages, WhatsApp promotions, first-trial offers, KOL posts, and review platforms manually.

**Future clinic matching engine**

- User selects treatment goal, budget, location, skin concern, and urgency.
- Ruvisa compares clinics by rating, price, promotion, distance, treatment type, trend signals, and user reviews.
- The AI agent explains trade-offs: best price, best quality, best-rated, nearest, or safest match.
- Users can discover products, routines, promotions, and clinics in one journey.
- Users can book directly from Ruvisa when a partner clinic or treatment place is available.

**Why this is powerful**

- Higher ticket size than product-only affiliate commerce.
- Clinics need qualified leads.
- Users need trust before booking beauty treatments.
- Hong Kong has strong beauty clinic culture and high willingness to spend on appearance.
- Ruvisa can organize scattered promotions and trial offers into transparent, comparable recommendations.

**Suggested visual:**  
Two-sided marketplace: users ↔ products + clinics, powered by Ruvisa AI matching.

**Speaker note:**  
This expands Ruvisa from an e-commerce tool into a platform. Product recommendation is the entry point; beauty treatment discovery is the larger monetization opportunity.

---

## Slide 8 — Competitor Analysis

### Ruvisa is not another beauty shop; it is the intelligence layer above beauty shopping.

| Competitor | Current strength | Gap Ruvisa solves |
| --- | --- | --- |
| **MindBeautyHK** | Hong Kong beauty and wellness booking platform with products and many beauty partners | Strong for booking and offers, but less focused on AI skin analysis, ingredient evidence, adaptive routine learning, and product-treatment matching from one skin profile |
| **@cosme / Cosme HK** | Trusted beauty reviews, rankings, awards, and consumer popularity signals | Useful for popularity, but rankings are not personalized to the user's selfie, budget, past purchases, or current skin progress |
| **Sasa** | Major Hong Kong beauty retailer with large product catalog, stores, discounts, and authenticity trust | Product-rich but still retail-led; users must decide by category, brand, promotion, and reviews rather than skin-state evidence |
| **iHerb** | Global health, wellness, supplement, skincare, and beauty marketplace with huge product range and reviews | Large catalog, but recommendations are broad and not connected to facial skin analysis, local HK clinics, or adaptive beauty routines |
| **Lookfantastic** | Premium international beauty retailer with large global catalog and brand selection | Strong commerce, but limited localized skin-progress tracking, clinic/treatment booking, and AI explanation around why a product fits the user's skin |
| **Beautylish** | Curated beauty retail, editorial content, makeup tools, and international shopping | Good for discovery and premium beauty, but not built around skin concern diagnosis or longitudinal skin improvement |
| **Strawberrynet** | Discount-driven international beauty shopping with wide product availability | Price appeal is strong, but users still compare deals manually and lack transparent skin-product fit |
| **Care to Beauty** | International pharmacy-style beauty retailer with skincare, sunscreen, and dermocosmetic brands | Strong product access, but not personalized by user image, local treatment need, or outcome history |

**Why Ruvisa is the solution**

- **Personalized starting point:** Ruvisa begins with the user's selfie and skin concern profile, not a generic product category.
- **Evidence-based matching:** Recommendations use ingredient evidence, review signals, skin type, price, and concern fit.
- **Adaptive learning:** Ruvisa learns from purchases and skin progress, so the next product or routine becomes more personalized over time.
- **Full beauty journey:** Ruvisa can recommend skincare, makeup, routines, promotions, and eventually clinics or treatments in one place.
- **Transparent decision support:** Users see why a product or treatment is recommended, helping them avoid hype-driven purchases and scattered promotions.

**Suggested visual:**  
Competitor map: product catalog depth vs personalization depth. Sasa/iHerb/Lookfantastic sit high on catalog depth; @cosme sits high on review trust; MindBeauty sits high on booking. Ruvisa sits at the intersection of AI personalization, product commerce, and treatment matching.

**Speaker note:**  
The competitors are strong, but they mostly solve separate parts of the journey: shopping, reviews, discounts, or booking. Ruvisa connects them through one skin-aware recommendation engine, so the user can move from "what is wrong with my skin?" to "what should I buy or book within my budget?"

---

## Slide 9 — Business Model

### Ruvisa has B2C and B2B revenue paths from products, routines, and clinics.

**Revenue streams**

1. **B2C product commission**
  Ruvisa earns commission when consumers purchase recommended skincare, makeup, or beauty products through the platform.
2. **B2C treatment booking commission**
  Ruvisa earns commission or booking fees when users book partner clinic treatments, first-trial offers, or beauty services.
3. **Sponsored but evidence-labeled placements**
  Brands and clinics can promote offers, but recommendation labels remain explainable, concern-matched, and transparent.
4. **Premium subscription**
  Users can pay for advanced progress tracking, routine optimization, ingredient conflict warnings, and deeper AI consultation.
5. **B2B beauty intelligence**
  Dashboards for brands, retailers, and clinics: trending concerns, budget demand, ingredient interest, review sentiment, and conversion insights.

**Why it can monetize**

- Users already spend on beauty.
- Recommendations influence purchase intent.
- Clinics have high customer acquisition value.
- Personalization increases retention and repeat engagement.
- A routine recommender creates recurring product discovery, not just one-time purchase intent.

**Suggested visual:**  
Revenue stream icons: product commission, subscription, clinic lead, brand dashboard.

**Speaker note:**  
Ruvisa is both B2C and B2B. Consumers get personalized guidance; brands and clinics get qualified intent. Ruvisa earns when users buy, book, subscribe, or when partners pay for verified visibility and insights.

---

## Slide 10 — Go-To-Market Strategy

### Win trust through KOLs, university communities, promotions, and free skin consultations.

**Phase 1: Hong Kong beta**

- Target students, young professionals, skincare beginners, and beauty enthusiasts aged 18-35.
- Launch through university networks, Instagram, TikTok, Threads, beauty forums, and micro-influencers.
- Use university ambassadors and student beauty communities to build early trust and consumer loyalty.
- Focus on acne, pigmentation, texture, pores, redness, and wrinkle concerns.

**Phase 2: KOL and trust-building**

- Run "scan your skin journey" campaigns.
- Encourage monthly progress scans.
- Publish ingredient explanation content.
- Collect before/after user stories and recommendation feedback.
- Partner with beauty influencers and KOLs for endorsement, demo videos, scan challenges, and routine reviews.

**Phase 3: Promotions and commercial partnerships**

- Approach indie skincare brands for affiliate or sample partnerships.
- Approach clinics for early lead-generation pilots.
- Add "best under budget" product and clinic recommendation campaigns.
- Offer free AI skin consultation campaigns to attract first-time users.
- Aggregate product discounts, clinic first-trial offers, and limited-time promotions so users can compare options in one app.
- Enable direct booking to available treatment places once partner supply is onboarded.

**Phase 4: Southeast Asia expansion**

- Localize catalog and clinic/treatment data for Singapore, Malaysia, Indonesia, Thailand, and the Philippines.
- Use marketplace trends from Shopee, Lazada, and TikTok Shop to update product discovery.
- Work with local KOLs and campus communities in each market before scaling paid acquisition.

**Suggested visual:**  
Funnel: KOL/uni campaign → free skin scan → recommendation → product purchase → clinic booking → progress tracking.

**Speaker note:**  
The go-to-market strategy starts with trust. Beauty purchases are emotional and social, so KOLs, student communities, and free skin consultations are stronger early channels than generic ads.

---

## Slide 11 — Milestone Plan And HKD 100,000 Use

### SEED funding will turn Ruvisa from launched prototype into measurable market pilot.


| Timeline | Quarter milestone | Target output |
| -------- | ----------------- | ------------- |
| Q1: Months 1-3 | Production hardening and Hong Kong beta launch | Stable domain, HTTPS, monitoring, analytics, improved onboarding, 100-200 registered users, 50+ completed scans, and university ambassador outreach |
| Q2: Months 4-6 | Recommendation validation and catalog expansion | Track product clicks, saves, purchases, user feedback, and routine engagement; expand Hong Kong products, ingredient mappings, review evidence, discounts, and promotions |
| Q3: Months 7-9 | Commercial pilot and clinic preparation | Start affiliate/brand tests, build clinic dataset with rating/price/promotion/treatment schema, run free AI consultation campaigns, and secure 2-3 brand or clinic pilot conversations |
| Q4: Months 10-12 | Market validation and Southeast Asia readiness | Measure retention, repeat scans, conversion, and booking intent; refine revenue model, prepare partner pitch materials, and plan first SEA catalog localization for Singapore or Malaysia |


**HKD 100,000 funding use**

- HKD 25,000 — AI product hardening: AI/LLM credits, model inference costs, hosting, monitoring, security, backups, and production reliability
- HKD 25,000 — data expansion: product catalog, ingredient evidence, review labeling, clinic dataset, and treatment/promotion data
- HKD 20,000 — beta user acquisition: KOL/student ambassador content, micro-influencer tests, and free AI skin consultation campaigns
- HKD 15,000 — UX/UI polish: mobile optimization, onboarding flow, demo assets, and recommendation explanation design
- HKD 10,000 — advisor/legal/privacy/compliance preparation
- HKD 5,000 — contingency

**Suggested visual:**  
12-month quarterly roadmap + funding allocation donut chart.

**Speaker note:**  
The funding is used to validate the market, not just build features. The goal is to produce measurable evidence of user demand, trust, and purchase behavior.

---

## Slide 12 — Revenue Forecast And Closing Ask

### Revenue target: reach HKD 100,000 annualized pilot revenue after validation.

**Pilot revenue forecast**


| Period                  | Revenue source                           | Conservative target    |
| ----------------------- | ---------------------------------------- | ---------------------- |
| Months 1-3              | Beta, no monetization focus              | HKD 0-5,000            |
| Months 4-6              | Affiliate clicks + early brand/KOL tests | HKD 10,000-20,000      |
| Months 7-9              | Product commission + premium users + clinic leads | HKD 25,000-35,000      |
| Months 10-12            | Product partnerships + clinic booking pilot | HKD 45,000-60,000      |
| **Year 1 pilot target** | Combined revenue                         | **HKD 80,000-120,000** |


**Key assumptions**

- 500-1,000 registered users in pilot year.
- 20-30% complete at least one scan.
- 5-10% click recommended products, promotions, or clinic leads.
- Revenue starts with affiliate and small partnership pilots, then grows through clinic lead generation and treatment booking commission.

**Funding ask**
Ruvisa is seeking **HKD 100,000 SEED support** to validate an AI-powered beauty commerce platform that can reduce product waste, improve recommendation trust, and unlock product and clinic revenue across Hong Kong and Southeast Asia.

**Closing line**
Ruvisa helps beauty consumers stop guessing and start choosing with evidence.

**Suggested visual:**  
Revenue ramp bar chart + final product mockup.

---

# Appendix — Source Notes

## Market Research Sources

- Hong Kong Consumer Council survey on skincare spending, unused products, and product waste: [https://www.consumer.org.hk/en/press-release/p-542-skincare-product-user-opinion-survey](https://www.consumer.org.hk/en/press-release/p-542-skincare-product-user-opinion-survey)
- South China Morning Post coverage of Hong Kong skincare spending and unused products: [https://www.scmp.com/news/hong-kong/hong-kong-economy/article/3159789/hongkongers-spend-average-hk3100-skincare-products](https://www.scmp.com/news/hong-kong/hong-kong-economy/article/3159789/hongkongers-spend-average-hk3100-skincare-products)
- Hong Kong beauty and personal care e-commerce market summary: [https://www.statista.com/outlook/emo/beauty-personal-care/hong-kong](https://www.statista.com/outlook/emo/beauty-personal-care/hong-kong)
- Hong Kong skincare market overview: [https://www.china-briefing.com/news/hong-kong-skincare-market-overview/](https://www.china-briefing.com/news/hong-kong-skincare-market-overview/)
- Southeast Asia skincare e-commerce market insights: [https://www.tmogroup.asia/insights/southeast-asia-skincare-ecommerce-market/](https://www.tmogroup.asia/insights/southeast-asia-skincare-ecommerce-market/)
- Southeast Asia health and beauty e-commerce market insights: [https://www.tmogroup.asia/insights/sea-health-beauty-ecommerce-market/](https://www.tmogroup.asia/insights/sea-health-beauty-ecommerce-market/)
- Southeast Asia beauty e-commerce and Shopee/TikTok Shop market discussion: [https://moojing-global.com/news-research/southeast-asia-beauty-ecommerce-shopee-2026](https://moojing-global.com/news-research/southeast-asia-beauty-ecommerce-shopee-2026)
- Social media influence on Southeast Asia skincare choices: [https://retailasia.com/news/how-social-media-redefining-skincare-choices-in-southeast-asia](https://retailasia.com/news/how-social-media-redefining-skincare-choices-in-southeast-asia)
- Shopee and POND'S AI beauty concierge example for online beauty choice overload: [https://ms.shopee.com/news/pond's-and-shopee-partner-to-deliver-marter-skincare-experiences-with-first-integrated-virtual-beauty-concierge-on-e-commerce/](https://ms.shopee.com/news/pond's-and-shopee-partner-to-deliver-marter-skincare-experiences-with-first-integrated-virtual-beauty-concierge-on-e-commerce/)
- Hong Kong Consumer Council beauty service warning on high-pressure sales: [https://www.consumer.org.hk/en/press-release/pretty-beauty](https://www.consumer.org.hk/en/press-release/pretty-beauty)
- Hong Kong beauty centre prepaid package and consumer complaint context: [https://www.scmp.com/news/hong-kong/article/3307491/hong-kong-beauty-centre-closure-sparks-probe-hk200000-prepaid-packages](https://www.scmp.com/news/hong-kong/article/3307491/hong-kong-beauty-centre-closure-sparks-probe-hk200000-prepaid-packages)
- Personalized skincare market growth and AI personalization context: [https://www.astuteanalytica.com/industry-report/personalized-skin-care-products-market](https://www.astuteanalytica.com/industry-report/personalized-skin-care-products-market)

## Competitor Research Sources

- MindBeautyHK beauty and wellness booking platform: [https://www.mindbeautyhk.com/](https://www.mindbeautyhk.com/)
- MindBeautyHK online store: [https://www.mindbeautyhk-store.com/](https://www.mindbeautyhk-store.com/)
- @cosme Hong Kong beauty review and ranking platform: [https://hk.cosme.net/](https://hk.cosme.net/)
- Sasa Hong Kong eShop: [https://www.sasa.com.hk/](https://www.sasa.com.hk/)
- Sasa Global eShop and company background: [https://buy.sasa.com/](https://buy.sasa.com/)
- iHerb beauty category and corporate overview: [https://www.iherb.com/c/beauty](https://www.iherb.com/c/beauty), [https://corporate.iherb.com/](https://corporate.iherb.com/)
- Lookfantastic company and online beauty retail overview: [https://www.lookfantastic.com.sg/info/about-us.list](https://www.lookfantastic.com.sg/info/about-us.list)
- Beautylish beauty retail platform: [https://www.beautylish.com/](https://www.beautylish.com/)
- Care to Beauty online cosmetics retailer: [https://www.caretobeauty.com/](https://www.caretobeauty.com/)

## Ruvisa Project Evidence

- Frontend: `frontend/src/App.jsx`, `frontend/src/api.js`
- Backend: `api/main.py`, `api/routes.py`
- Agentic AI: `agent/graph.py`, `agent/tools.py`, `agent/prompts.py`, `agent/llm.py`
- Ranking results: `docs/results_discussion_ranking_matching_revised.md`
- Skin analyzer results: `docs/results_discussion_skin_analyzer_revised.md`
- Product intelligence / DeBERTa evidence: `docs/deberta_cv_results_summary.json`
- Deployment notes: `DEPLOY.md`

