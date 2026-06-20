# 1.1 Background

Every person in this world wants to look better in terms of appearance from time to time. That is why the beauty market industry is predicted to be worth **USD 889.33 billion by 2027** [8]. Yet, online beauty retail and store platforms still heavily rely on constant search loops, marketing and product claims, unstructured product trends, and large amounts of unreliable user reviews, whereas clinically based skincare product guidance tends to exist on another platform or in offline clinics, which are often very costly and not reachable for all beauty product shoppers. Everyone keeps getting stuck in this search-and-product-claim loop, until it results in nearly **49% of users owning unopened facial skincare products for over a year**, which eventually creates financial waste and general waste [9]. There is also a lack of beauty e-commerce platforms that can provide transparent and personalized product recommendations for each user, creating a never-ending cycle of waste, marketing-claim bias, and ineffective product selection. Eventually, **70% of beauty consumers report being overwhelmed by too many product choices** [10], which shows how digital beauty platforms can still fail to give users what they really need.

This problem loop has motivated the creation of **Ruvisa**, which is a skincare e-commerce system that combines computer-vision skin analysis, structured product intelligence derived from INCI lists and user reviews, vector-based personalization, and an agentic conversational layer that grounds recommendations in user history and outcomes. Personalized skincare commerce requires more than accurate classifiers in isolation; it requires novel engineering that aligns multimodal signals, user feedback, and grounded dialogue in one deployable system [1].

# 1.2 Literature Review

## 1.2.1 Digital Skincare Commerce and Consumer-Facing Skin Analysis

Online skincare retail has shifted from static catalogs to more interactive experiences, where users increasingly expect guidance that reflects skin type, concerns, and budget rather than popularity or brand alone. At the same time, mobile imaging and accessible machine learning have normalized self-assessment tools that estimate texture, lesions, pigmentation, and aging-related patterns from facial images [1], [2]. However, many commercial systems still prioritize engagement or conversion rather than transparent reasoning grounded in formulation science and peer experience.

## 1.2.2 Computer Vision for Facial Skin Conditions

Research in this area spans lesion detection, severity grading, segmentation, and attribute estimation for conditions such as acne, pigmentation, and redness [2], [3]. Typical pipelines combine object detectors or segmentation models with crop-based or region-based classifiers, sometimes augmented with anatomical localization such as face parsing [2]. Despite promising progress, major challenges remain in robustness to lighting, pose, skin-tone variation, makeup, and domain shift between controlled clinical images and consumer photographs [2], [3]. Much of the literature also focuses on isolated benchmark performance rather than integration with downstream recommendation and longitudinal user feedback.

## 1.2.3 Text and Knowledge for Cosmetic Ingredients (INCI)

Ingredient lists provide a structured and ordered representation of product formulation, and recent studies have shown that ingredient-based modeling can support product effect prediction and personalized skincare recommendation [1], [7]. Because ingredient order carries formulation meaning, sequence-aware analysis is particularly useful for cosmetic reasoning [1]. Compared with surface-level claims text, ingredient-based representations are more directly tied to what a product actually contains and therefore provide a stronger foundation for explainable product intelligence [1], [7].

## 1.2.4 Reviews as Experiential Signal

User reviews provide another important source of information by capturing reported effectiveness, irritation, and suitability for different skin types. However, review language is noisy, subjective, and highly variable in detail. Prior research in aspect-based sentiment analysis shows that fine-grained opinion extraction from reviews remains challenging, especially when aspect mentions are sparse, implicit, or domain-specific [5]. This makes product-level aggregation a non-trivial step, particularly when the goal is to derive stable concern-specific signals rather than general sentiment alone.

## 1.2.5 Recommendation Beyond Collaborative Filtering

Traditional recommender systems are commonly categorized into collaborative filtering, content-based filtering, and hybrid approaches [4]. Hybrid recommender systems are especially valuable because they combine multiple signals to reduce weaknesses such as cold start and data sparsity [4]. In skincare recommendation, hybridization is particularly important because user skin state, ingredient evidence, review-derived outcomes, and purchasing history all describe different but complementary parts of the recommendation problem.

## 1.2.6 Large Language Models and Agentic Interfaces

Large language models have made conversational interfaces more flexible and natural, but they also introduce risks such as hallucination, stale reasoning, and weak grounding. ReAct-style frameworks address this by interleaving reasoning with external tool use, allowing the model to retrieve structured information and act over external systems instead of relying only on internal generation [6]. This is especially relevant in skincare recommendation, where accurate answers depend on current user state, product evidence, and recommendation logic rather than open-ended text generation alone.

# 1.3 Research Gaps

Despite progress in individual areas, several important gaps remain for integrated skincare e-commerce systems.

First, many existing works remain separated by **modality silos**. Vision systems often report lesion detection or severity results without aligning their outputs to the same concern schema later used for product representation and recommendation. This weakens end-to-end personalization.

Second, there is still **weak grounding in conversational retail systems**. LLM-based assistants may provide fluent answers, but they often lack strong binding to SKU-level evidence, structured ingredients, purchase history, and user outcomes, which increases hallucination risk and reduces actionability [6].

Third, many recommendation systems rely on **static personalization**. They may model user preference at one time point, but they do not explicitly re-rank products based on repeat skin scans, changing concern severity, or post-purchase outcomes within a unified framework [4].

Fourth, there remains a tension between **explainability and flexibility**. Purely neural systems may be difficult to audit, while rule-only systems may miss context and interaction effects. Hybrid approaches that remain interpretable at scale are still relatively under-documented as coherent deployed systems [1], [4], [7].

Finally, there is an **evaluation mismatch** between research and deployment. Many studies focus on one model component in isolation, whereas real-world skincare platforms require aligned multimodal pipelines operating under storage, latency, and deployment constraints [2], [3].

# 1.4 Objective of the Study

This project builds **Ruvisa** to address the gap between generic e-commerce search and trustworthy, personalized skincare guidance. In current digital beauty shopping, users rarely have access to a single system that connects what their skin shows in a photo, what a product formula supports through ingredient evidence, what users report in reviews, and how previous purchases and outcomes should influence the next recommendation.

The objective of this study is therefore to design, implement, and describe an end-to-end system that integrates multimodal skin analysis, hybrid product intelligence, aligned user-product matching, and grounded conversational assistance. More specifically, the project combines image-based skin concern estimation, ingredient lookup and position-aware evidence scoring, DeBERTa-based multi-label modeling, rule-based review aggregation, cosine-similarity matching with adaptive feedback, and a ReAct-style tool-using agent tied to live user and catalog state [1], [6]. In doing so, the study also aims to contribute a documented and deployable engineering architecture, including APIs, storage, artifacts, and pipelines, so that the approach is reproducible and auditable rather than only a collection of isolated model scores.

\newpage

# References

[1] J. Lee, H. Yoon, S. Kim, C. Lee, J. Lee, and S. Yoo, "Deep learning-based skin care product recommendation: A focus on cosmetic ingredient analysis and facial skin conditions," *Journal of Cosmetic Dermatology*, vol. 23, pp. 2066-2077, 2024. doi: 10.1111/jocd.16218. Available: <https://onlinelibrary.wiley.com/doi/10.1111/jocd.16218>

[2] Z. Li, K. C. Koban, T. L. Schenck, R. E. Giunta, Q. Li, and Y. Sun, "Artificial Intelligence in Dermatology Image Analysis: Current Developments and Future Trends," *Journal of Clinical Medicine*, vol. 11, no. 22, p. 6826, 2022. doi: 10.3390/jcm11226826. Available: <https://www.mdpi.com/2077-0383/11/22/6826>

[3] S. P. Choy et al., "Systematic review of deep learning image analyses for the diagnosis and monitoring of skin disease," *npj Digital Medicine*, vol. 6, p. 180, 2023. doi: 10.1038/s41746-023-00914-8. Available: <https://pmc.ncbi.nlm.nih.gov/articles/PMC10533565/>

[4] E. Cano and M. Morisio, "Hybrid Recommender Systems: A Systematic Literature Review," *Intelligent Data Analysis*, vol. 21, no. 6, pp. 1487-1524, 2017. doi: 10.3233/IDA-163209. Available: <https://iris.polito.it/retrieve/e384c42f-da94-d4b2-e053-9f05fe0a1d67/ErionCanoHRS-SLR.pdf>

[5] G. Brauwers and F. Frasincar, "A Survey on Aspect-Based Sentiment Classification," *ACM Computing Surveys*, 2021. doi: 10.1145/3503044. Available: <https://personal.eur.nl/frasincar/papers/CSUR2022/csur2022.pdf>

[6] S. Yao et al., "ReAct: Synergizing Reasoning and Acting in Language Models," in *ICLR*, 2023. Available: <https://arxiv.org/abs/2210.03629>

[7] C. Liu et al., "Beauty Beyond Words: Explainable Beauty Product Recommendations Using Ingredient-Based Product Attributes," 2024. Available: <https://arxiv.org/html/2409.13628v1>

[8] Global Growth Insights, "Beauty Market Growth Driven by 6.94% CAGR by 2035," 2026. Available: <https://www.globalgrowthinsights.com/market-reports/beauty-market-121581>

[9] Consumer Council, "Close to 50% Users Stockpiled Unused Skincare for Beyond a Year Easily Causing Wastage Equally Unwise to Purchase Full Product Range Based on Brand Hype," 2022. Available: <https://www.consumer.org.hk/en/press-release/p-542-skincare-product-user-opinion-survey>

[10] Healthline Media, *Rebuilding Consumer Trust in Skincare Report*. Available: <https://www.healthlinemedia.com/assets/files/HealthlineMedia_Rebuilding_Consumer_Trust_in_Skincare_Report_Final.pdf>

[11] C. Yu, J. Wang, C. Peng, C. Gao, G. Yu, and N. Sang, “BiSeNet: Bilateral Segmentation Network for Real-Time Semantic Segmentation,” in *Proc. Eur. Conf. Comput. Vis. (ECCV)*, 2018, pp. 325–341.

[12] C.-H. Lee, Z. Liu, L. Wu, and P. Luo, “MaskGAN: Towards Diverse and Interactive Facial Image Manipulation,” in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2020.

[13] G. Jocher, A. Chaurasia, and J. Qiu. (2023). *Ultralytics YOLOv8* [Online]. Available: <https://github.com/ultralytics/ultralytics>

[14] K. He, X. Zhang, S. Ren, and J. Sun, “Deep Residual Learning for Image Recognition,” in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)*, Las Vegas, NV, USA, 2016, pp. 770–778.

[15] O. Ronneberger, P. Fischer, and T. Brox, “U-Net: Convolutional Networks for Biomedical Image Segmentation,” in *Medical Image Computing and Computer-Assisted Intervention (MICCAI)*, 2015, pp. 234–241.

[16] X. Wu, L. Wen, G. Lin, Z. Ni, J. Suo, and Q. Tian, “Joint Acne Image Grading and Counting via Label Distribution Learning,” in *Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV)*, 2019, pp. 10642–10651.

[17] N. Hayashi, H. Akamatsu, M. Kawashima, et al., “Establishment of grading criteria for acne severity,” *J. Dermatol.*, vol. 35, no. 5, pp. 255–260, May 2008.

[18] T. Karras, S. Laine, and T. Aila, “A Style-Based Generator Architecture for Generative Adversarial Networks,” in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2019, pp. 4401–4410.

[19] J. Moon, H. Chung, and I. Jang, “Facial Wrinkle Segmentation for Cosmetic Dermatology: Pretraining with Texture Map-Based Weak Supervision,” in *Pattern Recognition*, ICPR 2024, 2025, pp. 319–334.

[20] R. Pascanu, T. Mikolov, and Y. Bengio, “On the difficulty of training recurrent neural networks,” in *Proc. 30th Int. Conf. Mach. Learn. (ICML)*, 2013, pp. 1310–1318.

[21] Z. Zheng, P. Wang, W. Liu, J. Li, R. Ye, and D. Ren, “Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression,” in *Proc. AAAI Conf. Artif. Intell.*, vol. 34, no. 7, 2020, pp. 12993–13000.

[22] X. Li, W. Wang, L. Wu, S. Chen, X. Hu, J. Li, J. Tang, and J. Yang, “Generalized Focal Loss: Learning Qualified and Distributed Bounding Boxes for Dense Object Detection,” in *Adv. Neural Inf. Process. Syst. (NeurIPS)*, vol. 33, 2020, pp. 21002–21012.

[23] A. Shrivastava, A. Gupta, and R. Girshick, “Training Region-Based Object Detectors with Online Hard Example Mining,” in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)*, Las Vegas, NV, USA, 2016, pp. 761–769.

[24] European Parliament and Council of the European Union, “Regulation (EC) No 1223/2009 of 30 November 2009 on cosmetic products,” *Off. J. Eur. Union*, vol. L 342, pp. 59–209, Dec. 2009.

[25] U.S. Food and Drug Administration. *Cosmetics Labeling Guide* [Online]. Available: <https://www.fda.gov/cosmetics/cosmetics-labeling-regulations/cosmetics-labeling-guide>

[26] INCIDecoder. *Decode your skincare ingredients* [Online]. Available: <https://incidecoder.com/>

[27] P. He, J. Gao, and W. Chen, “DeBERTaV3: Improving DeBERTa using ELECTRA-style pre-training with gradient-disentangled embedding sharing,” *arXiv preprint arXiv:2111.09543*, 2021. [Online]. Available: <https://arxiv.org/abs/2111.09543>

[28] P. He, X. Liu, J. Gao, and W. Chen, “DeBERTa: Decoding-enhanced BERT with disentangled attention,” in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2021.

[29] K. Clark, M.-T. Luong, Q. V. Le, and C. D. Manning, “ELECTRA: Pre-training text encoders as discriminators rather than generators,” in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2020.

[30] K. Sechidis, G. Tsoumakas, and I. Vlahavas, “On the stratification of multi-label data,” in *Mach. Learn. Knowl. Discov. Databases (ECML PKDD)*, 2011, pp. 145–158.

[31] P. Szymański and T. Kajdanowicz, “A Network Perspective on Stratification of Multi-Label Data,” in *Proc. Mach. Learn. Res. (PMLR)*, vol. 74, 2017, pp. 22–35.

[32] I. Loshchilov and F. Hutter, “Decoupled Weight Decay Regularization,” in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2019.

[33] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, “BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding,” in *Proc. North Amer. Chapter Assoc. Comput. Linguist. Hum. Lang. Technol. (NAACL-HLT)*, Minneapolis, MN, USA, 2019, pp. 4171–4186.

[34] Sephora Hong Kong. *Sephora Hong Kong* [Online]. Available: <https://www.sephora.hk/>

[35] K. Kritsakorn. *Acne Dataset*. Roboflow Universe [Online]. Available: <https://universe.roboflow.com/kritsakorn/acne-kbm0q>

[36] G. Salton, A. Wong, and C. S. Yang, “A Vector Space Model for Automatic Indexing,” *Commun. ACM*, vol. 18, no. 11, pp. 613–620, 1975.

[37] P. Jaccard, “Étude comparative de la distribution florale dans une portion des Alpes et du Jura,” *Bull. Soc. Vaudoise Sci. Nat.*, vol. 37, pp. 547–579, 1901.

[38] S. Ramírez. *FastAPI* [Online]. Available: <https://fastapi.tiangolo.com/>

[39] LangChain Inc. *LangGraph Overview* [Online]. Available: <https://docs.langchain.com/langgraph>

[40] Ollama. *API Introduction* [Online]. Available: <https://docs.ollama.com/api/introduction>
