# Ruvisa — IEEE Conference Paper: Conclusion (Draft)

> **Note.** Section number is a placeholder—renumber once merged with the full paper.

## VI. Conclusion

We presented **Ruvisa**, an end-to-end skincare e-commerce system built on a single thesis: a recommendation is trustworthy only when it is evidence-backed, personalized, and self-correcting *within one deployable system*. Ruvisa realizes this by projecting computer-vision skin analysis, position-aware INCI ingredient evidence, and review-derived effectiveness onto a shared seven-concern representation, fusing them into an explainable, knowledge-graph–aware recommender, and grounding a ReAct-style agent in each user's live state and outcome history—an early, evidence-grounded instance of agentic commerce.

Component-wise experiments on real data substantiate the design. Ingredient supervision proved a more faithful basis for concern inference than marketing claims (DeBERTa micro-F1 0.921 vs. 0.851); a highly reliable severity classifier (0.994 accuracy) combined with a deliberately recall-oriented detector still yields a coherent end-to-end concern representation; the adaptive layer re-ranks decisively from real outcomes despite touching few products; and the agent selected the correct tools in 100% of runs while tool grounding cut product hallucination by 38% relative. An end-to-end case study traced a single user from selfie to grounded conversational recommendation, demonstrating that the contributions compose rather than merely coexist.

Limitations remain—chiefly small-object lesion localization, residual agent hallucination under a small local model, and the absence of a formal user study. Our future work follows directly: strengthening localization, constraining agent generation to retrieved SKUs, conducting a longitudinal human-subject study of purchase satisfaction and skin outcomes, and extending the grounded tool interface toward autonomous, agent-initiated checkout to fully realize trustworthy agentic commerce in beauty retail.
