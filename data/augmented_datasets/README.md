# External Dataset Mappings: Domain Diversification

## Overview
To build a robust legal classifier, our model must understand risk across various legal domains and not just standard website terms. This directory contains augmented data sourced from world-class legal research repositories. By integrating these, we transform our classifier from a specialized tool into a universal legal risk detector.

---

## Datasets Integrated

### 1. OPP-115 (Online Privacy Policies)
This dataset provides deep insights into how companies handle user data. 
* **Focus:** Data security, third-party sharing, and user control.
* **Benefit:** Strengthens the model’s ability to detect subtle privacy violations.

### 2. CUAD (Contract Understanding Atticus Dataset)
Curated by legal experts, this corpus contains complex commercial contracts.
* **Focus:** Liability caps, termination rights, and professional obligations.
* **Benefit:** Teaches the model the "dense legalese" used in corporate transactions.

---

## The Mapping Strategy
Legal experts originally used dozens of specialized categories. To maintain consistency with our primary pipeline, we implemented a **Universal Risk Schema**.

### Our Logic:
| Original Category Example | Risk Label | Reason for Mapping |
| :--- | :--- | :--- |
| Data Security / Notice Period | **Good** | Empowers the user or protects their status. |
| Governing Law / Parties | **Neutral** | Standard boilerplate with no direct risk. |
| Third Party Sharing / Uncapped Liability | **Bad** | Direct threat to privacy or financial safety. |



---

## Technical Pipeline
1. **Automated Fetching:** Data is pulled directly from the Hugging Face Hub.
2. **Text Extraction:** Custom Regular Expressions (Regex) were designed to isolate the actual legal clauses from complex JSON structures.
3. **Risk Conflict Resolution:** For multi-labeled clauses, we implement a "Safety First" logic—if a clause contains even one risky element, it is flagged as **Bad**.

## Impact on Model Performance
By including these datasets, we perform **Cross-Domain Validation**. This proves that our AI actually understands the *concepts* of legal risk rather than just memorizing specific sentences from the training set.
