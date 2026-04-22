# Terms of Service Classifier: Exploratory Data Analysis

This document explains the first phase of this project. The goal was to explore the legal text data before training the AI model.

## Project Overview
Welcome to the data analysis phase of our natural language processing project. Nobody truly reads Terms of Service agreements because they are filled with confusing legal jargon. Our main goal is to build an artificial intelligence model that automatically reads these documents and highlights unfair or risky rules. 

This document explains our initial data exploration step. We analyzed thousands of legal clauses to understand their structure before training our model.

## Visualizations and Insights

### Class Balance Chart
We counted the total number of good, neutral, and bad clauses. The graph shows that neutral statements are the most common, which is expected in legal documents. The ratio is stable enough to train a highly accurate model.

![Class Distribution of TOS Clauses](results/1_class_distribution_of_TOS_clauses.png)

### Clause Length Histogram
We calculated the exact word count for every legal sentence. The chart visualizes the distribution of text length across different risk categories. This helps us see if bad clauses intentionally use longer and more complex wording to hide unfair rules.

![Clause Length Distribution by Risk Category](results/2_clause_length_distribution_by_risk_category.png)

### Vocabulary Word Clouds
We created visual word collages for every category. In these images, words that appear more frequently are drawn larger. This gives readers an immediate understanding of the tone. For example, the bad category prominently displays words related to cookies and external parties.

![Word Count Visualizations](results/3_word_count.jpg)

### Top Vocabulary Extraction
We mathematically extracted the most common single words and two-word phrases. We discovered that bad clauses frequently use phrases regarding prior notice and data sharing. This proves to our team that the model must learn context, not just simple keywords.

**Top Identified Risk Phrases (Bad Category):**
```text
- personal data:          793 occurrences
- party cookies:          335 occurrences
- prior notice:           233 occurrences
- tracking pixels:        228 occurrences
- personal information:   227 occurrences
- parties involved:       216 occurrences
- data parties:           208 occurrences
- cookies used:           184 occurrences
- involved operation:     182 occurrences
- device fingerprinting:  178 occurrences
- web beacons:            176 occurrences
- service uses:           169 occurrences

## Conclusion
This exploration proves our dataset is clean, well structured, and ready for the next phase. The insights gathered here will guide how we teach our model to spot dangerous legal terms.
