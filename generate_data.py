import numpy as np
import pandas as pd
import os

def generate_credit_data(num_samples=2000, random_seed=42):
    np.random.seed(random_seed)
    
    # 1. Age: 18 to 75, normally distributed around 40
    age = np.random.normal(loc=42, scale=12, size=num_samples).astype(int)
    age = np.clip(age, 18, 75)
    
    # 2. Annual Income: $15,000 to $200,000, log-normal distribution to be realistic
    annual_income = np.random.lognormal(mean=10.8, sigma=0.5, size=num_samples)
    annual_income = np.clip(annual_income, 15000, 200000).astype(int)
    
    # 3. Employment Status: Employed, Self-Employed, Unemployed, Retired
    # Probabilities depend on age
    employment_status = []
    for a in age:
        if a < 22:
            status = np.random.choice(['Employed', 'Self-Employed', 'Unemployed'], p=[0.4, 0.1, 0.5])
        elif a > 65:
            status = np.random.choice(['Retired', 'Employed', 'Self-Employed'], p=[0.8, 0.1, 0.1])
        else:
            status = np.random.choice(['Employed', 'Self-Employed', 'Unemployed'], p=[0.75, 0.18, 0.07])
        employment_status.append(status)
    employment_status = np.array(employment_status)
    
    # 4. Number of Credit Accounts: 1 to 20, correlated with age
    num_accounts = []
    for a in age:
        max_accounts = min(20, int(a / 2.5) + 3)
        n = np.random.randint(1, max_accounts)
        num_accounts.append(n)
    num_accounts = np.array(num_accounts)
    
    # 5. Credit History Length (in years): correlated with age
    credit_history_length = []
    for i, a in enumerate(age):
        max_history = a - 18
        if max_history <= 0:
            hist = np.random.uniform(0.1, 1.0)
        else:
            hist = np.random.uniform(1.0, max_history)
        credit_history_length.append(round(hist, 1))
    credit_history_length = np.array(credit_history_length)
    
    # 6. Credit Utilization Ratio: 0.0 to 1.2
    # High utilization correlated with lower income and unemployment
    credit_util_ratio = []
    for i in range(num_samples):
        income_factor = (200000 - annual_income[i]) / 200000
        emp = employment_status[i]
        emp_factor = 0.3 if emp == 'Unemployed' else 0.0
        base = np.random.beta(a=2, b=5) # peak around 0.28
        ratio = base + 0.3 * income_factor + emp_factor
        credit_util_ratio.append(round(np.clip(ratio, 0.0, 1.2), 2))
    credit_util_ratio = np.array(credit_util_ratio)
    
    # 7. Number of Late Payments: 0 to 15, correlated with credit utilization and employment status
    num_late_payments = []
    for i in range(num_samples):
        util = credit_util_ratio[i]
        emp = employment_status[i]
        
        # Base probability of late payments
        lambda_val = util * 3.0
        if emp == 'Unemployed':
            lambda_val += 2.0
            
        late = np.random.poisson(lam=lambda_val)
        num_late_payments.append(min(15, late))
    num_late_payments = np.array(num_late_payments)
    
    # 8. Payment History: percentage of on-time payments (0.5 to 1.0)
    # Strongly inversely related to number of late payments and credit utilization
    payment_history = []
    for i in range(num_samples):
        late = num_late_payments[i]
        util = credit_util_ratio[i]
        if late == 0:
            hist = np.random.uniform(0.98, 1.0)
        else:
            hist = 1.0 - (late * 0.05) - (util * 0.05)
        payment_history.append(round(np.clip(hist, 0.5, 1.0), 2))
    payment_history = np.array(payment_history)
    
    # 9. Loan Amount: $1,000 to $100,000, correlated with income
    loan_amount = []
    for inc in annual_income:
        max_loan = min(100000, int(inc * 0.8))
        min_loan = 1000
        if max_loan <= min_loan:
            loan = min_loan
        else:
            loan = np.random.randint(min_loan, max_loan)
        loan_amount.append(loan)
    loan_amount = np.array(loan_amount)
    
    # 10. Existing Debts: $0 to $150,000, correlated with income
    existing_debts = []
    for inc in annual_income:
        max_debt = min(150000, int(inc * 1.2))
        debt = np.random.uniform(0, max_debt)
        existing_debts.append(int(debt))
    existing_debts = np.array(existing_debts)
    
    # 11. Debt-to-Income Ratio: (Existing Debts + Loan Amount) / Annual Income
    # We clip it to make it realistic
    debt_to_income = []
    for i in range(num_samples):
        # Monthly debt payments estimate: 1% of existing debt + 2% of new loan (approximation)
        monthly_debt = (existing_debts[i] * 0.01) + (loan_amount[i] * 0.02)
        monthly_income = annual_income[i] / 12.0
        dti = monthly_debt / monthly_income
        debt_to_income.append(round(np.clip(dti, 0.02, 0.95), 2))
    debt_to_income = np.array(debt_to_income)
    
    # 12. Loan Purpose: Debt Consolidation, Home Improvement, Education, Auto, Business, Major Purchase
    purposes = ['Debt Consolidation', 'Home Improvement', 'Education', 'Auto', 'Business', 'Major Purchase']
    loan_purpose = np.random.choice(purposes, size=num_samples, p=[0.35, 0.20, 0.15, 0.10, 0.12, 0.08])
    
    # --- Generate Target Variable: Creditworthy ---
    # We define a scoring index
    # Normalize features to [0, 1] range for scoring
    norm_income = annual_income / 200000.0
    norm_late = num_late_payments / 15.0
    norm_util = credit_util_ratio / 1.2
    norm_history_len = credit_history_length / 40.0
    norm_pay_history = (payment_history - 0.5) / 0.5
    norm_dti = debt_to_income / 0.95
    
    # Employment multipliers
    emp_scores = {'Employed': 1.0, 'Self-Employed': 0.8, 'Retired': 0.85, 'Unemployed': 0.2}
    emp_score = np.array([emp_scores[e] for e in employment_status])
    
    # Calculate latent creditworthiness score
    # High score means more creditworthy
    latent_score = (
        0.15 * norm_income +
        0.15 * norm_history_len +
        0.25 * norm_pay_history +
        0.15 * emp_score -
        0.15 * norm_late -
        0.15 * norm_util -
        0.15 * norm_dti
    )
    
    # Add logistic noise to make it probabilistic
    noise = np.random.logistic(loc=0, scale=0.08, size=num_samples)
    final_score = latent_score + noise
    
    # Threshold for creditworthiness (e.g., 50th-60th percentile)
    threshold = np.percentile(final_score, 40) # ~60% creditworthy (1), 40% not creditworthy (0)
    creditworthy = (final_score >= threshold).astype(int)
    
    # Build dataframe
    df = pd.DataFrame({
        'Age': age,
        'Annual Income': annual_income,
        'Employment Status': employment_status,
        'Loan Amount': loan_amount,
        'Existing Debts': existing_debts,
        'Credit Utilization Ratio': credit_util_ratio,
        'Number of Credit Accounts': num_accounts,
        'Payment History': payment_history,
        'Credit History Length': credit_history_length,
        'Number of Late Payments': num_late_payments,
        'Debt-to-Income Ratio': debt_to_income,
        'Loan Purpose': loan_purpose,
        'Creditworthy': creditworthy
    })
    
    return df

if __name__ == '__main__':
    print("Generating synthetic credit dataset...")
    df = generate_credit_data(num_samples=2500)
    
    output_path = 'credit_data.csv'
    df.to_csv(output_path, index=False)
    print(f"Dataset generated successfully and saved to {output_path}")
    print(f"Data shape: {df.shape}")
    print(f"Class distribution:\n{df['Creditworthy'].value_counts(normalize=True)}")
    print("\nSample Data:")
    print(df.head())
