"""
Synthetic Dataset Generator for B2B Deal Intelligence Research Prototype.
Generates realistic pre-offer CPQ/CRM deal records and expert evaluation responses.
"""

import json
import csv
import random
from pathlib import Path

def generate_sample_data(num_records: int = 150, seed: int = 42):
    random.seed(seed)
    
    industries = ["Logistics", "Manufacturing", "Healthcare", "Financial Services", "Retail", "Software"]
    regions = ["North America", "EMEA", "APAC", "LATAM"]
    sales_channels = ["Direct", "Partner", "Inside Sales"]
    product_categories = ["Standard Modular", "Enterprise Custom", "Cloud Platform", "Hardware Hybrid"]
    
    records = []
    expert_evaluations = []
    
    for i in range(1, num_records + 1):
        case_id = f"DEAL-{i:03d}"
        
        # 1. CORE FEATURE GROUPS (Pre-offer information ONLY)
        # PRICE Features
        price_total_amount = round(random.uniform(50000, 750000), 2)
        price_discount_percentage = round(random.uniform(0.05, 0.40), 3)
        price_payment_terms_days = random.choice([30, 60, 90, 120])
        price_margin_percentage = round(random.uniform(0.15, 0.55), 3)
        
        # PRODUCT Features
        product_complexity_score = round(random.uniform(1.0, 10.0), 1)
        product_config_count = random.randint(1, 25)
        product_category = random.choice(product_categories)
        product_is_customized = 1 if product_complexity_score > 6.0 else 0
        
        # ORGANIZATION Features
        organization_sales_region = random.choice(regions)
        organization_team_size = random.randint(2, 12)
        organization_partner_involved = random.choice([0, 1])
        organization_sales_channel = random.choice(sales_channels)
        
        # CUSTOMER Features
        customer_industry = random.choice(industries)
        customer_company_size_employees = random.randint(100, 15000)
        customer_annual_revenue_millions = round(random.uniform(10.0, 2000.0), 1)
        customer_existing_client = random.choice([0, 1])
        
        # 2. TARGET OUTCOME (Win/Loss deterministic rule with noise)
        win_score = (
            (0.35 * (1.0 - price_discount_percentage / 0.40)) +
            (0.25 * (1.0 if customer_existing_client == 1 else 0.4)) +
            (0.20 * (1.0 - product_complexity_score / 10.0)) +
            (0.20 * (price_margin_percentage / 0.55)) +
            random.uniform(-0.15, 0.15)
        )
        win_loss_outcome = 1 if win_score >= 0.52 else 0
        
        record = {
            "case_id": case_id,
            "timestamp": f"2026-08-{random.randint(1,28):02d}T10:00:00Z",
            # Price Group
            "price_total_amount": price_total_amount,
            "price_discount_percentage": price_discount_percentage,
            "price_payment_terms_days": price_payment_terms_days,
            "price_margin_percentage": price_margin_percentage,
            # Product Group
            "product_complexity_score": product_complexity_score,
            "product_config_count": product_config_count,
            "product_category": product_category,
            "product_is_customized": product_is_customized,
            # Organization Group
            "organization_sales_region": organization_sales_region,
            "organization_team_size": organization_team_size,
            "organization_partner_involved": organization_partner_involved,
            "organization_sales_channel": organization_sales_channel,
            # Customer Group
            "customer_industry": customer_industry,
            "customer_company_size_employees": customer_company_size_employees,
            "customer_annual_revenue_millions": customer_annual_revenue_millions,
            "customer_existing_client": customer_existing_client,
            # Target Label
            "win_loss_outcome": win_loss_outcome,
        }
        records.append(record)
        
        # 3. EXPERT EVALUATION RECORD (For a subset of cases)
        if i <= 30:  # 30 detailed expert cases
            expert_id = f"EXPERT-{((i - 1) % 3) + 1:02d}"  # 3 experts
            
            # Simulated Expert prediction with noise
            expert_prob = min(max(win_score + random.uniform(-0.10, 0.10), 0.05), 0.95)
            expert_pred = 1 if expert_prob >= 0.50 else 0
            
            # Expert Q1-Q5 answers
            exp_eval = {
                "case_id": case_id,
                "expert_id": expert_id,
                "q1_b2b_experience": random.choice(["3-10 years", "11+ years"]),
                "q2_ai_tool_usage": random.choice([2, 3, 4]),
                "q3_predicting_experience": "Regularly predicts deal win probability in CRM",
                "q4_information_sufficiency": random.choice(["Yes", "Maybe"]),
                "q5_assessed_outcome": "WIN" if expert_pred == 1 else "LOSS",
                "expert_predicted_prob": round(expert_prob, 2),
                "expert_confidence": round(random.uniform(0.70, 0.95), 2),
                "expert_important_factors": [
                    "price_discount_percentage" if price_discount_percentage > 0.25 else "price_total_amount",
                    "customer_existing_client" if customer_existing_client == 1 else "product_complexity_score",
                    "organization_partner_involved" if organization_partner_involved == 1 else "organization_team_size",
                ],
                "expert_explanation": f"Based on {customer_industry} market conditions and discount of {price_discount_percentage*100:.1f}%.",
                # Usefulness rating (1 to 7 scale)
                "rating_prediction_alone": random.choice([3, 4, 4, 5]),
                "rating_prediction_and_explanation": random.choice([5, 6, 6, 7]),
            }
            expert_evaluations.append(exp_eval)

    # Save to data/raw/
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    json_path = raw_dir / "deals_dataset.json"
    with open(json_path, "w") as f:
        json.dump(records, f, indent=2)
        
    csv_path = raw_dir / "deals_dataset.csv"
    if records:
        keys = records[0].keys()
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(records)

    # Save to data/experts/
    expert_dir = Path("data/experts")
    expert_dir.mkdir(parents=True, exist_ok=True)
    
    exp_path = expert_dir / "expert_evaluations.json"
    with open(exp_path, "w") as f:
        json.dump(expert_evaluations, f, indent=2)

    print(f"Generated {len(records)} raw deal records at {csv_path}")
    print(f"Generated {len(expert_evaluations)} expert evaluation records at {exp_path}")

if __name__ == "__main__":
    generate_sample_data()
