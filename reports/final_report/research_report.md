# 🔬 B2B DEAL INTELLIGENCE RESEARCH REPORT
**Generated Date:** 2026-09-29 18:00:24 UTC  
**Primary Metric:** Balanced Accuracy  

---

## 1. DATASET SUMMARY
- **Source File:** `/Users/saikumar/.gemini/antigravity-ide/scratch/deal_intelligence/data/raw/deals_dataset.csv`
- **Total Ingested Records:** 150
- **Raw Win Outcome Count:** 90
- **Raw Loss Outcome Count:** 60

## 2. DATA PROCESSING SUMMARY
- **Initial Input Records:** 150
- **Duplicates Detected & Removed:** 0
- **Invalid Rows Removed:** 0
- **Post-Offer Leaked Fields Stripped:** None
- **Final Valid Dataset Size:** 150

## 3. FEATURE GROUP SUMMARY
- **Total Encoded Features:** 29
- **PRICE Group Features Count:** 4
- **PRODUCT Group Features Count:** 7
- **ORGANIZATION Group Features Count:** 9
- **CUSTOMER Group Features Count:** 9

## 4. PRICE FEATURES
- price_total_amount, price_discount_percentage, price_payment_terms_days, price_margin_percentage

## 5. PRODUCT FEATURES
- product_complexity_score, product_config_count, product_category_Cloud Platform, product_category_Enterprise Custom, product_category_Hardware Hybrid, product_category_Standard Modular, product_is_customized

## 6. ORGANIZATION FEATURES
- organization_sales_region_APAC, organization_sales_region_EMEA, organization_sales_region_LATAM, organization_sales_region_North America, organization_team_size, organization_partner_involved, organization_sales_channel_Direct, organization_sales_channel_Inside Sales, organization_sales_channel_Partner

## 7. CUSTOMER FEATURES
- customer_industry_Financial Services, customer_industry_Healthcare, customer_industry_Logistics, customer_industry_Manufacturing, customer_industry_Retail, customer_industry_Software, customer_company_size_employees, customer_annual_revenue_millions, customer_existing_client

## 8. TRAINING / TEST METHODOLOGY
- **Train/Test Split:** 80% Train / 20% Test (Stratified)
- **Random State Seed:** 42 (Reproducible)

## 9. XGBOOST CONFIGURATION
- **Algorithm:** XGBoost Classifier (or HistGradientBoosting fallback)
- **Hyperparameters:** `n_estimators=100`, `max_depth=4`, `learning_rate=0.05`, `random_state=42`

## 10. BALANCED ACCURACY (PRIMARY METRIC)
$$\text{Balanced Accuracy} = \frac{1}{2} \left( \frac{\text{TP}}{\text{TP} + \text{FN}} + \frac{\text{TN}}{\text{TN} + \text{FP}} \right) = \mathbf{0.7361}$$

## 11. CONFUSION MATRIX
```
                 Predicted WIN    Predicted LOSS
Actual WIN       13              5              
Actual LOSS      3               9              
```

## 12. PRECISION
- **Precision (WIN):** 0.8125

## 13. RECALL
- **Overall Recall:** 0.7222

## 14. F1-SCORE
- **F1-Score:** 0.7647

## 15. WIN / LOSS RECALL DISTRIBUTION
- **WIN Recall (TP / (TP + FN)):** 0.7222
- **LOSS Recall (TN / (TN + FP)):** 0.7500

## 16. SHAP GLOBAL FEATURE GROUP IMPORTANCE
- **CUSTOMER**: Total SHAP = 1.6652 (39.27% importance across 9 features)
- **PRICE**: Total SHAP = 1.5515 (36.59% importance across 4 features)
- **PRODUCT**: Total SHAP = 0.8237 (19.42% importance across 7 features)
- **ORGANIZATION**: Total SHAP = 0.2001 (4.72% importance across 9 features)
- **OTHER**: Total SHAP = 0.0000 (0.0% importance across 0 features)


## 17. SHAP PER-CASE EXPLANATION SAMPLES
- **Case DEAL-002 [EXPERT-02]**: Expert=WIN vs AI=LOSS | Agreement=False | Summary: Case DEAL-002 [EXPERT-02 vs AI]: Predictions DISAGREED (WIN vs LOSS). Agreed Factors: [customer_existing_client]. Expert-Only: [organization_partner_involved, price_total_amount]. AI-Identified: [price_margin_percentage, price_discount_percentage, product_complexity_score, product_config_count].
- **Case DEAL-005 [EXPERT-02]**: Expert=WIN vs AI=LOSS | Agreement=False | Summary: Case DEAL-005 [EXPERT-02 vs AI]: Predictions DISAGREED (WIN vs LOSS). Agreed Factors: [product_complexity_score]. Expert-Only: [organization_team_size, price_total_amount]. AI-Identified: [customer_existing_client, customer_industry_Manufacturing, price_margin_percentage, product_config_count].
- **Case DEAL-010 [EXPERT-01]**: Expert=WIN vs AI=WIN | Agreement=True | Summary: Case DEAL-010 [EXPERT-01 vs AI]: Predictions AGREED (WIN vs WIN). Agreed Factors: [customer_existing_client]. Expert-Only: [organization_team_size, price_total_amount]. AI-Identified: [price_discount_percentage, product_complexity_score, customer_industry_Manufacturing, price_margin_percentage].


## 18. SALES EXPERT PREDICTIONS
- Preserved independent Sales Expert predictions across evaluated cases.
- **Note:** Expert predictions were strictly excluded from model feature matrix to prevent label leakage.

## 19. Q1–Q5 QUESTIONNAIRE RESULTS SCHEMA
- Q1 (B2B Experience), Q2 (AI Tool Usage), Q3 (Predictive Experience), Q4 (Information Sufficiency), Q5 (Outcome Assessment) implemented and preserved per expert.

## 20. EXPERT VS AI COMPARISON
- Evaluated agreement/disagreement per case.
- Identified shared factors, expert-only factors, and AI-discovered factors without forcing artificial consensus.

## 21. PREDICTION ALONE USEFULNESS
- **Mean Rating:** 3.97 / 7.00

## 22. PREDICTION + EXPLANATION USEFULNESS
- **Mean Rating:** 5.93 / 7.00

## 23. DIFFERENCE BETWEEN CONDITIONS
- **Usefulness Lift (Prediction + Explanation vs. Prediction Alone):** +1.97 points
- **Conclusion:** Evaluated 30 case ratings from 3 experts. Prediction Alone Mean Rating: 3.97/7.00 | Prediction + Explanation Mean Rating: 5.93/7.00. Usefulness Lift: +1.97 points (Hypothesis Supported).

## 24. LIMITATIONS
- Synthetic dataset generated for initial pipeline validation; real production deployment requires expanding real CRM/CPQ historical deal records.
- Expert sample size (N=30) will grow as additional sales reps complete Q1-Q5 evaluation sessions.

---
*Report generated automatically by Deal Intelligence Agent Research Engine.*
