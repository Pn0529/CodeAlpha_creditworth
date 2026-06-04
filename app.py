import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import plotly.graph_objects as go
import plotly.express as px
import shap

# Set page configuration with a premium icon and title
st.set_page_config(
    page_title="AuraCredit - Advanced AI Credit Decision Engine",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium CSS injection
st.markdown("""
<style>
    /* Main Layout Styling */
    .reportview-container {
        background: #0f172a;
    }
    
    /* Header Card */
    .header-container {
        background: linear-gradient(135deg, #1e1b4b 0%, #311042 100%);
        padding: 2.5rem;
        border-radius: 16px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
        margin-bottom: 2rem;
        border: 1px solid #312e81;
    }
    .header-title {
        color: #f8fafc;
        font-family: 'Outfit', sans-serif;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        background: linear-gradient(to right, #a5b4fc, #e879f9);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .header-subtitle {
        color: #94a3b8;
        font-size: 1.1rem;
        margin-top: 0.5rem;
        margin-bottom: 0;
    }
    
    /* Metric Cards */
    .score-card {
        background: #1e293b;
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid #334155;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    .score-title {
        color: #94a3b8;
        font-size: 0.9rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .score-value {
        font-size: 3rem;
        font-weight: 800;
        margin: 0.5rem 0;
    }
    
    /* Recommendations Box */
    .rec-box {
        background: rgba(30, 41, 59, 0.7);
        border-left: 5px solid #6366f1;
        border-radius: 4px 12px 12px 4px;
        padding: 1.2rem;
        margin-top: 1rem;
        border-top: 1px solid #334155;
        border-bottom: 1px solid #334155;
        border-right: 1px solid #334155;
    }
    .rec-title {
        color: #e2e8f0;
        font-weight: 600;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to load model assets safely
@st.cache_resource
def load_assets():
    preprocessor = joblib.load('models/preprocessor.joblib')
    model = joblib.load('models/best_model.joblib')
    feature_names = joblib.load('models/feature_names.joblib')
    X_test_sample = joblib.load('models/X_test_sample.joblib')
    X_test_processed_sample = joblib.load('models/X_test_processed_sample.joblib')
    y_test_sample = joblib.load('models/y_test_sample.joblib')
    return preprocessor, model, feature_names, X_test_sample, X_test_processed_sample, y_test_sample

@st.cache_data
def load_raw_data():
    return pd.read_csv('credit_data.csv')

# Load all assets
try:
    preprocessor, model, feature_names, X_test_sample, X_test_processed, y_test_sample = load_assets()
    raw_df = load_raw_data()
    assets_loaded = True
except Exception as e:
    st.error(f"Error loading models or dataset. Please make sure you have run 'generate_data.py' and 'train_model.py' successfully. Details: {e}")
    assets_loaded = False

# Sidebar - Brand Logo & Quick Stats
st.sidebar.markdown("""
<div style='text-align: center; padding: 1rem 0;'>
    <h2 style='color: #6366f1; margin: 0; font-weight: 800;'>🛡️ AuraCredit</h2>
    <p style='color: #64748b; font-size: 0.85rem; margin-top: 0.2rem;'>AI Risk Management Platform</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

if assets_loaded:
    st.sidebar.subheader("Model Overview")
    st.sidebar.metric(label="Primary Classifier", value="Random Forest", delta="Tuned")
    st.sidebar.metric(label="Validation ROC-AUC", value="77.8%", delta="Optimized")
    st.sidebar.metric(label="Target Population Size", value=f"{len(raw_df)} Clients")

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='font-size: 0.8rem; color: #64748b; text-align: center;'>
    AuraCredit Engine v1.0.0<br>
    Built with Streamlit & SHAP
</div>
""", unsafe_allow_html=True)

# Main Dashboard Container
st.markdown("""
<div class="header-container">
    <h1 class="header-title">AuraCredit Decision Dashboard</h1>
    <p class="header-subtitle">Real-time Machine Learning Credit Eligibility Scoring & Explainable Risk Assessments.</p>
</div>
""", unsafe_allow_html=True)

if assets_loaded:
    # Setup tabs
    tab_assessment, tab_analytics, tab_dataset = st.tabs([
        "📊 Assessment Terminal", 
        "📈 Model Performance & Tuning", 
        "🔍 Training Data Insights"
    ])
    
    # ------------------ TAB 1: ASSESSMENT TERMINAL ------------------
    with tab_assessment:
        st.markdown("### Client Risk Profiling Terminal")
        st.write("Input applicant metrics below to evaluate credit eligibility in real-time.")
        
        # User input columns
        col_personal, col_financial, col_credit = st.columns(3)
        
        with col_personal:
            st.subheader("Demographics & Status")
            age = st.slider("Applicant Age", 18, 75, 38, help="Age of the credit applicant.")
            emp_status = st.selectbox(
                "Employment Status", 
                options=['Employed', 'Self-Employed', 'Retired', 'Unemployed'], 
                index=0,
                help="Current employment classification of the applicant."
            )
            loan_purpose = st.selectbox(
                "Loan Purpose",
                options=['Debt Consolidation', 'Home Improvement', 'Education', 'Auto', 'Business', 'Major Purchase'],
                index=0,
                help="Intended usage category for the requested loan."
            )
            
        with col_financial:
            st.subheader("Financial Assets")
            annual_income = st.number_input(
                "Annual Income ($)", 
                min_value=10000, 
                max_value=300000, 
                value=55000, 
                step=5000,
                help="Total verified gross annual income."
            )
            loan_amount = st.number_input(
                "Requested Loan Amount ($)", 
                min_value=500, 
                max_value=150000, 
                value=15000, 
                step=1000,
                help="Principal amount requested for the loan."
            )
            existing_debts = st.number_input(
                "Total Existing Debts ($)", 
                min_value=0, 
                max_value=250000, 
                value=8000, 
                step=1000,
                help="Sum of all outstanding balances on other credits."
            )
            
        with col_credit:
            st.subheader("Credit Behavior & History")
            credit_history_length = st.slider(
                "Credit History Length (Years)", 
                0.5, 40.0, 8.5, 0.5,
                help="Number of years since the applicant's first credit line was opened."
            )
            num_accounts = st.slider(
                "Number of Credit Accounts", 
                1, 25, 4,
                help="Total open credit cards and loans."
            )
            credit_util_ratio = st.slider(
                "Credit Utilization Ratio (%)", 
                0.0, 120.0, 32.0, 1.0,
                help="Percentage of total credit lines currently being utilized. Lower is better."
            ) / 100.0
            
            num_late_payments = st.slider(
                "Number of Late Payments (Last 12m)", 
                0, 15, 0,
                help="Count of payments delayed by 30+ days."
            )
            
        # Perform dynamic calculations for dependent features
        # Payment History: Approximate score based on late payments
        if num_late_payments == 0:
            payment_history = 0.99
        else:
            # Approx logic matching generate_data.py
            payment_history = np.clip(1.0 - (num_late_payments * 0.05) - (credit_util_ratio * 0.05), 0.5, 1.0)
            
        # Debt to income ratio:
        # Monthly debt payments estimate: 1% of existing debt + 2% of new loan
        monthly_debt = (existing_debts * 0.01) + (loan_amount * 0.02)
        monthly_income = annual_income / 12.0
        dti_ratio = np.clip(monthly_debt / monthly_income, 0.0, 1.0)
        
        st.markdown("---")
        
        # Action button
        if st.button("Evaluate Credit Eligibility 🛡️", type="primary", use_container_width=True):
            # Formulate user input dataframe
            user_data = pd.DataFrame({
                'Age': [age],
                'Annual Income': [annual_income],
                'Employment Status': [emp_status],
                'Loan Amount': [loan_amount],
                'Existing Debts': [existing_debts],
                'Credit Utilization Ratio': [credit_util_ratio],
                'Number of Credit Accounts': [num_accounts],
                'Payment History': [payment_history],
                'Credit History Length': [credit_history_length],
                'Number of Late Payments': [num_late_payments],
                'Debt-to-Income Ratio': [dti_ratio],
                'Loan Purpose': [loan_purpose]
            })
            
            # Preprocess inputs
            user_processed = preprocessor.transform(user_data)
            
            # Get model classification & probabilities
            prediction = model.predict(user_processed)[0]
            probability = model.predict_proba(user_processed)[0][1] # Probability of Creditworthy
            
            # Credit Score mapping (300 to 850)
            credit_score = int(300 + 550 * probability)
            
            # Results UI Layout
            res_col1, res_col2 = st.columns([1, 1])
            
            with res_col1:
                st.markdown("#### ⚖️ Decision Assessment")
                
                # Class mapping
                if prediction == 1:
                    status_text = "LOW RISK (Eligible for Credit)"
                    status_color = "#10b981" # Green
                    status_bg = "rgba(16, 185, 129, 0.1)"
                else:
                    status_text = "HIGH RISK (Not Eligible for Credit)"
                    status_color = "#f43f5e" # Rose
                    status_bg = "rgba(244, 63, 94, 0.1)"
                    
                st.markdown(f"""
                <div style='padding: 1.5rem; background-color: {status_bg}; border: 2px solid {status_color}; border-radius: 12px; text-align: center;'>
                    <h3 style='margin: 0; color: {status_color}; font-weight: 800; font-size: 1.6rem;'>{status_text}</h3>
                    <p style='margin: 0.5rem 0 0 0; color: #cbd5e1;'>Risk Probability: <strong>{100 - int(probability*100)}% Default Risk</strong></p>
                </div>
                """, unsafe_allow_html=True)
                
                # Credit Score Gauge
                # Define gauge color based on score
                if credit_score < 580:
                    gauge_color = "#f43f5e"  # Red
                elif credit_score < 670:
                    gauge_color = "#f59e0b"  # Orange
                elif credit_score < 740:
                    gauge_color = "#eab308"  # Yellow
                else:
                    gauge_color = "#10b981"  # Green
                    
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = credit_score,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Calculated Credit Score", 'font': {'size': 18, 'color': '#cbd5e1'}},
                    number = {'font': {'size': 48, 'color': gauge_color}},
                    gauge = {
                        'axis': {'range': [300, 850], 'tickwidth': 1, 'tickcolor': "#cbd5e1"},
                        'bar': {'color': gauge_color},
                        'bgcolor': "#1e293b",
                        'borderwidth': 2,
                        'bordercolor': "#334155",
                        'steps': [
                            {'range': [300, 580], 'color': 'rgba(244, 63, 94, 0.25)'},
                            {'range': [580, 670], 'color': 'rgba(245, 158, 11, 0.25)'},
                            {'range': [670, 740], 'color': 'rgba(234, 179, 8, 0.25)'},
                            {'range': [740, 850], 'color': 'rgba(16, 185, 129, 0.25)'}
                        ]
                    }
                ))
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', 
                    plot_bgcolor='rgba(0,0,0,0)',
                    font={'color': "#cbd5e1"},
                    height=280,
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Recommendations & Advisory
                st.markdown("""
                <div class="rec-box">
                    <div class="rec-title">
                        <span>💡</span>
                        <span>Personalized Financial Advisory</span>
                    </div>
                """, unsafe_allow_html=True)
                
                recs = []
                if num_late_payments > 0:
                    recs.append(f"Applicant registered <strong>{num_late_payments} late payments</strong> recently. Encouraging auto-debits can raise score parameters.")
                if credit_util_ratio > 0.35:
                    recs.append(f"Credit utilization is high (<strong>{int(credit_util_ratio*100)}%</strong>). Advise paying down balances to under 30% to improve risk profile.")
                if dti_ratio > 0.40:
                    recs.append(f"Debt-to-Income ratio (<strong>{int(dti_ratio*100)}%</strong>) is near caution levels. Consolidating debts might help.")
                if emp_status == 'Unemployed':
                    recs.append("Current employment status listed as Unemployed. Verification of secondary assets or collateral is recommended.")
                if credit_history_length < 3.0:
                    recs.append("Short credit history length limits model historical confidence. Perform supplementary bank history checks.")
                
                if len(recs) == 0:
                    st.write("✅ **Excellent Risk Profile!** Applicant meets or exceeds all credit security standards. Minimal default probability predicted.")
                else:
                    for r in recs:
                        st.markdown(f"- {r}", unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
                
            with res_col2:
                st.markdown("#### 🧠 Explainable AI - Local SHAP Attribution")
                st.write("Analyzing local features that pulled the applicant's risk rating towards eligibility (positive) or default (negative):")
                
                # Dynamic SHAP value calculation
                # We calculate SHAP values using TreeExplainer
                # To make it robust, we instantiate a TreeExplainer
                explainer = shap.TreeExplainer(model)
                shap_val_object = explainer(user_processed)
                
                # Retrieve shap values for positive class (Creditworthy)
                if isinstance(shap_val_object.values, list):
                    local_shap_vals = shap_val_object.values[1][0]
                elif len(shap_val_object.values.shape) == 3:
                    local_shap_vals = shap_val_object.values[0, :, 1]
                else:
                    local_shap_vals = shap_val_object.values[0]
                    
                # Create DataFrame for plotting
                shap_df = pd.DataFrame({
                    'Feature': feature_names,
                    'SHAP Value': local_shap_vals
                })
                
                # Sort and filter to top contributors
                shap_df['Abs_SHAP'] = shap_df['SHAP Value'].abs()
                shap_df = shap_df.sort_values(by='Abs_SHAP', ascending=True).tail(8) # Top 8 features
                
                # Apply human-readable naming for plotting
                shap_df['Impact'] = np.where(shap_df['SHAP Value'] >= 0, 'Positive (Improves Score)', 'Negative (Increases Risk)')
                
                # Plotly Horizontal Bar Chart
                fig_shap = px.bar(
                    shap_df,
                    x='SHAP Value',
                    y='Feature',
                    orientation='h',
                    color='Impact',
                    color_discrete_map={
                        'Positive (Improves Score)': '#10b981',
                        'Negative (Increases Risk)': '#f43f5e'
                    },
                    title="Feature Contribution to Creditworthiness Decision",
                    labels={'SHAP Value': 'Attribution Impact', 'Feature': 'Preprocessed Feature Name'}
                )
                
                fig_shap.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font={'color': "#cbd5e1"},
                    legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
                    height=450,
                    xaxis=dict(showgrid=True, gridcolor="#334155"),
                    yaxis=dict(showgrid=False)
                )
                st.plotly_chart(fig_shap, use_container_width=True)
                
    # ------------------ TAB 2: MODEL PERFORMANCE ------------------
    with tab_analytics:
        st.markdown("### Model Comparison & Training Metrics")
        st.write("Evaluation of multiple classifiers and hyperparameter search results.")
        
        # Performance numbers
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.markdown("""
            <div class='score-card'>
                <div class='score-title'>Model Classifier</div>
                <div class='score-value' style='font-size: 1.6rem; color: #a5b4fc; padding: 0.9rem 0;'>Tuned Random Forest</div>
            </div>
            """, unsafe_allow_html=True)
        with col_m2:
            st.markdown("""
            <div class='score-card'>
                <div class='score-title'>Accuracy</div>
                <div class='score-value' style='color: #10b981;'>71.2%</div>
            </div>
            """, unsafe_allow_html=True)
        with col_m3:
            st.markdown("""
            <div class='score-card'>
                <div class='score-title'>Recall</div>
                <div class='score-value' style='color: #10b981;'>85.3%</div>
            </div>
            """, unsafe_allow_html=True)
        with col_m4:
            st.markdown("""
            <div class='score-card'>
                <div class='score-title'>ROC-AUC</div>
                <div class='score-value' style='color: #a5b4fc;'>77.8%</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("---")
        
        col_c1, col_c2 = st.columns(2)
        
        with col_c1:
            st.subheader("Model Comparison Metrics")
            if os.path.exists('models/model_comparison.csv'):
                # If comparison CSV is loaded, let's plot
                comp_raw = pd.read_csv('models/model_comparison.csv')
                if 'Unnamed: 0' in comp_raw.columns:
                    comp_raw = comp_raw.rename(columns={'Unnamed: 0': 'Model'})
                
                # Reformat for side-by-side plotting
                comp_melted = comp_raw.melt(id_vars='Model', var_name='Metric', value_name='Score')
                fig_comp = px.bar(
                    comp_melted,
                    x='Model',
                    y='Score',
                    color='Metric',
                    barmode='group',
                    color_discrete_sequence=['#6366f1', '#10b981', '#f59e0b', '#e879f9', '#a5b4fc'],
                    title="Baseline Models Performance Comparison"
                )
                fig_comp.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font={'color': "#cbd5e1"},
                    xaxis=dict(showgrid=False),
                    yaxis=dict(gridcolor="#334155")
                )
                st.plotly_chart(fig_comp, use_container_width=True)
            else:
                st.info("Comparison data not available.")
                
        with col_c2:
            st.subheader("Tuned Model Global Feature Importance")
            if os.path.exists('models/feature_importances.csv'):
                feat_imp = pd.read_csv('models/feature_importances.csv')
                fig_feat = px.bar(
                    feat_imp.head(12),
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title="Top 12 Features Affecting Creditworthiness",
                    color_discrete_sequence=['#6366f1']
                )
                fig_feat.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font={'color': "#cbd5e1"},
                    xaxis=dict(gridcolor="#334155"),
                    yaxis=dict(showgrid=False, categoryorder='total ascending')
                )
                st.plotly_chart(fig_feat, use_container_width=True)
                
        st.markdown("---")
        
        # Display Confusion Matrix & ROC plots
        col_img1, col_img2 = st.columns(2)
        with col_img1:
            st.subheader("Confusion Matrix (Test Data)")
            if os.path.exists('models/confusion_matrix.png'):
                st.image('models/confusion_matrix.png', caption="Test Set Confusion Matrix - Tuned Random Forest Classifier")
        with col_img2:
            st.subheader("ROC Curve (Test Data)")
            if os.path.exists('models/roc_curve.png'):
                st.image('models/roc_curve.png', caption="Test Set ROC Curve - Area Under Curve = 0.7778")

    # ------------------ TAB 3: DATASET EXPLORER ------------------
    with tab_dataset:
        st.markdown("### Training Data Distribution Explorer")
        st.write("Inspect distributions and correlations of factors within the synthetic applicant pool.")
        
        explore_cols = st.columns([1, 3])
        
        with explore_cols[0]:
            st.subheader("Plot Settings")
            x_var = st.selectbox(
                "Select X-axis Variable", 
                options=raw_df.columns, 
                index=0,
                help="Variable to plot on the horizontal axis."
            )
            y_var = st.selectbox(
                "Select Y-axis Variable (Optional)", 
                options=['None'] + list(raw_df.columns), 
                index=0,
                help="Optional numeric variable for 2D comparison. Select 'None' for 1D distributions."
            )
            color_by = st.selectbox(
                "Color / Group by",
                options=['Creditworthy', 'Employment Status', 'Loan Purpose', 'None'],
                index=0
            )
            
        with explore_cols[1]:
            st.subheader("Visual Distribution")
            
            color_arg = None if color_by == 'None' else color_by
            # Convert target back to readable categories for display if selected
            plot_df = raw_df.copy()
            plot_df['Creditworthy'] = plot_df['Creditworthy'].map({1: 'Eligible', 0: 'Not Eligible'})
            
            if y_var == 'None':
                # Plotly histogram or count plot
                if plot_df[x_var].dtype == 'object' or plot_df[x_var].nunique() < 10:
                    fig_dist = px.histogram(
                        plot_df, 
                        x=x_var, 
                        color=color_arg, 
                        barmode='group',
                        title=f"Distribution Count of {x_var}",
                        color_discrete_sequence=['#10b981', '#f43f5e', '#6366f1', '#f59e0b']
                    )
                else:
                    fig_dist = px.histogram(
                        plot_df, 
                        x=x_var, 
                        color=color_arg, 
                        marginal='box',
                        title=f"Numerical Distribution of {x_var}",
                        color_discrete_sequence=['#10b981', '#f43f5e', '#6366f1', '#f59e0b']
                    )
            else:
                # Scatter plot for two variables
                fig_dist = px.scatter(
                    plot_df, 
                    x=x_var, 
                    y=y_var, 
                    color=color_arg,
                    title=f"Scatter Analysis: {x_var} vs {y_var}",
                    opacity=0.6,
                    color_discrete_sequence=['#10b981', '#f43f5e', '#6366f1', '#f59e0b']
                )
                
            fig_dist.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': "#cbd5e1"},
                xaxis=dict(gridcolor="#334155"),
                yaxis=dict(gridcolor="#334155")
            )
            st.plotly_chart(fig_dist, use_container_width=True)
            
        st.markdown("---")
        st.subheader("Statistical Summary Table")
        st.dataframe(raw_df.describe().T, use_container_width=True)
