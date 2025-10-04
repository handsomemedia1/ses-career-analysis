"""
Advanced Solutions for Multicollinearity in SES and Career Choice Analysis
Author: Research Team
Date: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# Load the analyzed data
df = pd.read_csv('analyzed_data_with_composites.csv')

print("="*70)
print("ADVANCED SOLUTIONS FOR MULTICOLLINEARITY")
print("="*70)

# ============================================================================
# SOLUTION 1: HIERARCHICAL REGRESSION (RECOMMENDED)
# ============================================================================

def hierarchical_regression(df):
    """
    Build models sequentially to see unique contribution of each predictor
    This is the BEST approach for publication
    """
    print("\n" + "="*70)
    print("SOLUTION 1: HIERARCHICAL REGRESSION")
    print("="*70)
    print("\nBuilding models sequentially to examine incremental variance explained")
    
    # Prepare data
    reg_data = df[['Career_Satisfaction', 'SES_Composite', 'Parental_Influence_Score',
                   'Gender_Influence_Score', 'Career_Autonomy', 'Gender']].dropna()
    
    y = reg_data['Career_Satisfaction']
    
    # MODEL 1: Demographics only
    X1 = reg_data[['Gender']]
    X1_const = sm.add_constant(X1)
    model1 = sm.OLS(y, X1_const).fit()
    
    print("\n--- MODEL 1: Demographics Only ---")
    print(f"R² = {model1.rsquared:.4f}")
    print(f"Adj. R² = {model1.rsquared_adj:.4f}")
    print(f"F-statistic = {model1.fvalue:.3f}, p = {model1.f_pvalue:.4f}")
    
    # MODEL 2: Add SES (objective measure)
    X2 = reg_data[['Gender', 'SES_Composite']]
    X2_const = sm.add_constant(X2)
    model2 = sm.OLS(y, X2_const).fit()
    
    print("\n--- MODEL 2: Demographics + SES ---")
    print(f"R² = {model2.rsquared:.4f}")
    print(f"Adj. R² = {model2.rsquared_adj:.4f}")
    print(f"ΔR² = {model2.rsquared - model1.rsquared:.4f}")
    print(f"F-statistic = {model2.fvalue:.3f}, p = {model2.f_pvalue:.4f}")
    
    # MODEL 3: Add Gender Influence (social constraints)
    X3 = reg_data[['Gender', 'SES_Composite', 'Gender_Influence_Score']]
    X3_const = sm.add_constant(X3)
    model3 = sm.OLS(y, X3_const).fit()
    
    print("\n--- MODEL 3: Demographics + SES + Gender Influence ---")
    print(f"R² = {model3.rsquared:.4f}")
    print(f"Adj. R² = {model3.rsquared_adj:.4f}")
    print(f"ΔR² = {model3.rsquared - model2.rsquared:.4f}")
    print(f"F-statistic = {model3.fvalue:.3f}, p = {model3.f_pvalue:.4f}")
    
    # MODEL 4: Add Career Autonomy (psychological factor)
    X4 = reg_data[['Gender', 'SES_Composite', 'Gender_Influence_Score', 'Career_Autonomy']]
    X4_const = sm.add_constant(X4)
    model4 = sm.OLS(y, X4_const).fit()
    
    print("\n--- MODEL 4: Full Model WITHOUT Parental Influence ---")
    print(f"R² = {model4.rsquared:.4f}")
    print(f"Adj. R² = {model4.rsquared_adj:.4f}")
    print(f"ΔR² = {model4.rsquared - model3.rsquared:.4f}")
    print(f"F-statistic = {model4.fvalue:.3f}, p = {model4.f_pvalue:.4f}")
    
    print("\n" + "-"*70)
    print("DETAILED RESULTS FOR MODEL 4:")
    print(model4.summary())
    
    # Create comparison table
    comparison = pd.DataFrame({
        'Model': ['1: Demographics', '2: + SES', '3: + Gender Influence', '4: + Career Autonomy'],
        'R²': [model1.rsquared, model2.rsquared, model3.rsquared, model4.rsquared],
        'Adj. R²': [model1.rsquared_adj, model2.rsquared_adj, model3.rsquared_adj, model4.rsquared_adj],
        'ΔR²': [model1.rsquared, 
                model2.rsquared - model1.rsquared,
                model3.rsquared - model2.rsquared,
                model4.rsquared - model3.rsquared],
        'F': [model1.fvalue, model2.fvalue, model3.fvalue, model4.fvalue],
        'p': [model1.f_pvalue, model2.f_pvalue, model3.f_pvalue, model4.f_pvalue]
    })
    
    print("\n--- HIERARCHICAL REGRESSION COMPARISON TABLE ---")
    print(comparison.to_string(index=False))
    
    return model4, comparison

# ============================================================================
# SOLUTION 2: SEPARATE MODELS FOR EACH PREDICTOR
# ============================================================================

def separate_models(df):
    """
    Run separate regression models to isolate each predictor's effect
    """
    print("\n" + "="*70)
    print("SOLUTION 2: SEPARATE MODELS")
    print("="*70)
    print("\nRunning separate models to examine independent effects")
    
    reg_data = df[['Career_Satisfaction', 'SES_Composite', 'Parental_Influence_Score',
                   'Gender_Influence_Score', 'Career_Autonomy', 'Gender']].dropna()
    
    y = reg_data['Career_Satisfaction']
    
    predictors = ['SES_Composite', 'Parental_Influence_Score', 'Gender_Influence_Score', 'Career_Autonomy']
    
    results = []
    
    for pred in predictors:
        # Simple regression with just this predictor + gender control
        X = reg_data[['Gender', pred]]
        X_const = sm.add_constant(X)
        model = sm.OLS(y, X_const).fit()
        
        # Extract coefficients for the predictor (not gender)
        coef = model.params[pred]
        se = model.bse[pred]
        t = model.tvalues[pred]
        p = model.pvalues[pred]
        ci_lower, ci_upper = model.conf_int().loc[pred]
        
        # Standardized coefficient
        X_std = StandardScaler().fit_transform(reg_data[[pred]])
        y_std = StandardScaler().fit_transform(y.values.reshape(-1, 1)).flatten()
        X_std_with_intercept = sm.add_constant(X_std)
        model_std = sm.OLS(y_std, X_std_with_intercept).fit()
        beta = model_std.params[1]
        
        results.append({
            'Predictor': pred.replace('_', ' '),
            'B': coef,
            'SE': se,
            't': t,
            'p': p,
            '95% CI Lower': ci_lower,
            '95% CI Upper': ci_upper,
            'β (standardized)': beta,
            'R²': model.rsquared,
            'Adj. R²': model.rsquared_adj
        })
        
        print(f"\n{pred}:")
        print(f"  B = {coef:.3f}, SE = {se:.3f}, t = {t:.3f}, p = {p:.4f}")
        print(f"  95% CI [{ci_lower:.3f}, {ci_upper:.3f}]")
        print(f"  β = {beta:.3f}, R² = {model.rsquared:.3f}")
    
    results_df = pd.DataFrame(results)
    print("\n--- SUMMARY TABLE ---")
    print(results_df.to_string(index=False))
    
    return results_df

# ============================================================================
# SOLUTION 3: MEDIATION ANALYSIS
# ============================================================================

def mediation_analysis(df):
    """
    Test if Parental Influence mediates the SES-Career Satisfaction relationship
    This addresses multicollinearity by modeling the causal pathway
    """
    print("\n" + "="*70)
    print("SOLUTION 3: MEDIATION ANALYSIS")
    print("="*70)
    print("\nTesting: SES → Parental Influence → Career Satisfaction")
    
    med_data = df[['Career_Satisfaction', 'SES_Composite', 'Parental_Influence_Score']].dropna()
    
    # Step 1: Total effect (c path): X → Y
    X = sm.add_constant(med_data['SES_Composite'])
    y = med_data['Career_Satisfaction']
    model_c = sm.OLS(y, X).fit()
    c = model_c.params['SES_Composite']
    c_p = model_c.pvalues['SES_Composite']
    
    print("\nStep 1: Total Effect (c path)")
    print(f"  SES → Career Satisfaction: B = {c:.3f}, p = {c_p:.4f}")
    
    # Step 2: Effect on mediator (a path): X → M
    M = med_data['Parental_Influence_Score']
    model_a = sm.OLS(M, X).fit()
    a = model_a.params['SES_Composite']
    a_p = model_a.pvalues['SES_Composite']
    
    print("\nStep 2: Effect on Mediator (a path)")
    print(f"  SES → Parental Influence: B = {a:.3f}, p = {a_p:.4f}")
    
    # Step 3: Direct effect (c' path) and mediator effect (b path): X + M → Y
    X_M = sm.add_constant(med_data[['SES_Composite', 'Parental_Influence_Score']])
    model_b = sm.OLS(y, X_M).fit()
    c_prime = model_b.params['SES_Composite']
    c_prime_p = model_b.pvalues['SES_Composite']
    b = model_b.params['Parental_Influence_Score']
    b_p = model_b.pvalues['Parental_Influence_Score']
    
    print("\nStep 3: Direct Effect (c' path) and Mediator Effect (b path)")
    print(f"  SES → Career Satisfaction (controlling for Parental Influence): B = {c_prime:.3f}, p = {c_prime_p:.4f}")
    print(f"  Parental Influence → Career Satisfaction: B = {b:.3f}, p = {b_p:.4f}")
    
    # Calculate indirect effect
    indirect_effect = a * b
    
    print("\n--- MEDIATION SUMMARY ---")
    print(f"Total Effect (c): {c:.3f}")
    print(f"Direct Effect (c'): {c_prime:.3f}")
    print(f"Indirect Effect (a*b): {indirect_effect:.3f}")
    print(f"Proportion Mediated: {(indirect_effect/c)*100:.1f}%")
    
    if abs(c_prime) < abs(c) and a_p < 0.05:
        if c_prime_p > 0.05:
            print("\nConclusion: FULL MEDIATION - Parental Influence fully mediates SES effect")
        else:
            print("\nConclusion: PARTIAL MEDIATION - Parental Influence partially mediates SES effect")
    else:
        print("\nConclusion: NO MEDIATION - Parental Influence does not mediate SES effect")
    
    return {'c': c, 'a': a, 'b': b, 'c_prime': c_prime, 'indirect': indirect_effect}

# ============================================================================
# SOLUTION 4: PRINCIPAL COMPONENTS REGRESSION
# ============================================================================

def principal_components_regression(df):
    """
    Use PCA to create orthogonal components, eliminating multicollinearity
    """
    print("\n" + "="*70)
    print("SOLUTION 4: PRINCIPAL COMPONENTS REGRESSION")
    print("="*70)
    
    reg_data = df[['Career_Satisfaction', 'SES_Composite', 'Parental_Influence_Score',
                   'Gender_Influence_Score', 'Career_Autonomy']].dropna()
    
    y = reg_data['Career_Satisfaction']
    X = reg_data.drop('Career_Satisfaction', axis=1)
    
    # Standardize predictors
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply PCA
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)
    
    print("\nPrincipal Components Variance Explained:")
    for i, var in enumerate(pca.explained_variance_ratio_):
        print(f"  PC{i+1}: {var:.3f} ({var*100:.1f}%)")
    
    print(f"\nCumulative variance (first 3 PCs): {pca.explained_variance_ratio_[:3].sum():.3f}")
    
    # Regression with first 3 components
    X_pca_3 = X_pca[:, :3]
    X_pca_const = sm.add_constant(X_pca_3)
    model_pca = sm.OLS(y, X_pca_const).fit()
    
    print("\n--- PC REGRESSION RESULTS ---")
    print(f"R² = {model_pca.rsquared:.4f}")
    print(f"Adj. R² = {model_pca.rsquared_adj:.4f}")
    print(f"F = {model_pca.fvalue:.3f}, p = {model_pca.f_pvalue:.4f}")
    
    print("\nComponent Loadings:")
    loadings = pd.DataFrame(
        pca.components_[:3].T,
        columns=['PC1', 'PC2', 'PC3'],
        index=X.columns
    )
    print(loadings.round(3))
    
    return model_pca, loadings

# ============================================================================
# SOLUTION 5: RIDGE REGRESSION
# ============================================================================

def ridge_regression(df):
    """
    Ridge regression penalizes large coefficients, handling multicollinearity
    """
    print("\n" + "="*70)
    print("SOLUTION 5: RIDGE REGRESSION")
    print("="*70)
    
    reg_data = df[['Career_Satisfaction', 'SES_Composite', 'Parental_Influence_Score',
                   'Gender_Influence_Score', 'Career_Autonomy', 'Gender']].dropna()
    
    y = reg_data['Career_Satisfaction'].values
    X = reg_data.drop('Career_Satisfaction', axis=1).values
    
    # Standardize
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
    
    # Try different alpha values
    alphas = [0.01, 0.1, 1, 10, 100]
    
    print("\nTesting different regularization strengths (alpha):")
    
    for alpha in alphas:
        ridge = Ridge(alpha=alpha)
        ridge.fit(X_scaled, y_scaled)
        score = ridge.score(X_scaled, y_scaled)
        
        print(f"\nAlpha = {alpha}:")
        print(f"  R² = {score:.4f}")
        print(f"  Coefficients: {ridge.coef_.round(3)}")
    
    # Use optimal alpha (alpha=1 is common)
    ridge_final = Ridge(alpha=1)
    ridge_final.fit(X_scaled, y_scaled)
    
    coef_df = pd.DataFrame({
        'Variable': reg_data.drop('Career_Satisfaction', axis=1).columns,
        'Ridge Coefficient': ridge_final.coef_
    })
    
    print("\n--- FINAL RIDGE COEFFICIENTS (alpha=1) ---")
    print(coef_df.to_string(index=False))
    
    return ridge_final, coef_df

# ============================================================================
# SOLUTION 6: USE ONLY NON-COLLINEAR PREDICTORS
# ============================================================================

def simplified_model(df):
    """
    Drop Parental_Influence_Score and use only SES_Composite
    This is the simplest solution
    """
    print("\n" + "="*70)
    print("SOLUTION 6: SIMPLIFIED MODEL (Remove Collinear Predictors)")
    print("="*70)
    
    reg_data = df[['Career_Satisfaction', 'SES_Composite', 'Gender_Influence_Score',
                   'Career_Autonomy', 'Gender']].dropna()
    
    y = reg_data['Career_Satisfaction']
    X = reg_data[['Gender', 'SES_Composite', 'Gender_Influence_Score', 'Career_Autonomy']]
    X_const = sm.add_constant(X)
    
    model = sm.OLS(y, X_const).fit()
    
    print("\nModel WITHOUT Parental_Influence_Score:")
    print(model.summary())
    
    # Check VIF
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    vif_data = pd.DataFrame()
    vif_data["Variable"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    
    print("\n--- VIF CHECK (should all be < 10) ---")
    print(vif_data)
    
    return model, vif_data

# ============================================================================
# VISUALIZATION: COMPARE ALL APPROACHES
# ============================================================================

def visualize_comparisons(hier_comparison, separate_results, mediation_results):
    """
    Create visualization comparing different approaches
    """
    print("\n" + "="*70)
    print("CREATING COMPARISON VISUALIZATIONS")
    print("="*70)
    
    # Figure 1: Hierarchical R² progression
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    ax1 = axes[0]
    models = hier_comparison['Model'].values
    r_squared = hier_comparison['R²'].values
    delta_r = hier_comparison['ΔR²'].values
    
    x_pos = np.arange(len(models))
    bars = ax1.bar(x_pos, r_squared, color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12'], alpha=0.7)
    
    # Add ΔR² labels
    for i, (bar, delta) in enumerate(zip(bars, delta_r)):
        if i > 0:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'ΔR²={delta:.3f}', ha='center', fontsize=9, fontweight='bold')
    
    ax1.set_xlabel('Model', fontweight='bold')
    ax1.set_ylabel('R²', fontweight='bold')
    ax1.set_title('Hierarchical Regression: R² Progression', fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([m.split(':')[0] for m in models], rotation=45, ha='right')
    ax1.set_ylim(0, max(r_squared) * 1.2)
    ax1.grid(axis='y', alpha=0.3)
    
    # Figure 2: Separate models comparison
    ax2 = axes[1]
    predictors = separate_results['Predictor'].values
    betas = separate_results['β (standardized)'].values
    p_values = separate_results['p'].values
    
    colors = ['green' if p < 0.05 else 'gray' for p in p_values]
    
    y_pos = np.arange(len(predictors))
    bars = ax2.barh(y_pos, betas, color=colors, alpha=0.7)
    
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(predictors)
    ax2.set_xlabel('Standardized Coefficient (β)', fontweight='bold')
    ax2.set_title('Separate Models: Individual Predictor Effects', fontweight='bold')
    ax2.axvline(0, color='black', linestyle='-', linewidth=0.5)
    ax2.grid(axis='x', alpha=0.3)
    
    # Add significance markers
    for i, (bar, p) in enumerate(zip(bars, p_values)):
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        x_pos = bar.get_width() + 0.01 if bar.get_width() > 0 else bar.get_width() - 0.01
        ax2.text(x_pos, bar.get_y() + bar.get_height()/2, sig,
                ha='left' if bar.get_width() > 0 else 'right', va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('09_multicollinearity_solutions.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: 09_multicollinearity_solutions.png")
    plt.close()
    
    # Figure 3: Mediation diagram
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Draw mediation paths
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    
    # Boxes
    box_props = dict(boxstyle='round,pad=0.3', facecolor='lightblue', edgecolor='black', linewidth=2)
    
    ax.text(0.1, 0.5, 'SES\nComposite', ha='center', va='center', fontsize=12, 
            fontweight='bold', bbox=box_props, transform=ax.transAxes)
    
    ax.text(0.5, 0.8, 'Parental\nInfluence', ha='center', va='center', fontsize=12, 
            fontweight='bold', bbox=box_props, transform=ax.transAxes)
    
    ax.text(0.9, 0.5, 'Career\nSatisfaction', ha='center', va='center', fontsize=12, 
            fontweight='bold', bbox=box_props, transform=ax.transAxes)
    
    # Arrows with coefficients
    arrow_props = dict(arrowstyle='->', lw=2, color='black')
    
    # a path
    ax.annotate('', xy=(0.42, 0.75), xytext=(0.18, 0.55),
                arrowprops=arrow_props, transform=ax.transAxes)
    ax.text(0.25, 0.68, f"a = {mediation_results['a']:.3f}", fontsize=10, 
            fontweight='bold', transform=ax.transAxes)
    
    # b path
    ax.annotate('', xy=(0.82, 0.55), xytext=(0.58, 0.75),
                arrowprops=arrow_props, transform=ax.transAxes)
    ax.text(0.70, 0.68, f"b = {mediation_results['b']:.3f}", fontsize=10, 
            fontweight='bold', transform=ax.transAxes)
    
    # c' path (direct effect)
    ax.annotate('', xy=(0.82, 0.5), xytext=(0.18, 0.5),
                arrowprops=dict(arrowstyle='->', lw=2, color='red', linestyle='--'), 
                transform=ax.transAxes)
    ax.text(0.5, 0.42, f"c' = {mediation_results['c_prime']:.3f}\n(direct effect)", 
            fontsize=10, fontweight='bold', ha='center', color='red', transform=ax.transAxes)
    
    # Indirect effect
    ax.text(0.5, 0.92, f"Indirect Effect = {mediation_results['indirect']:.3f}", 
            fontsize=11, fontweight='bold', ha='center', 
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5),
            transform=ax.transAxes)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('Mediation Analysis: Does Parental Influence Mediate SES Effect?',
                fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('10_mediation_diagram.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: 10_mediation_diagram.png")
    plt.close()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run all multicollinearity solutions"""
    
    print("\n" + "="*70)
    print("COMPREHENSIVE SOLUTIONS FOR MULTICOLLINEARITY")
    print("="*70 + "\n")
    
    # Solution 1: Hierarchical Regression (BEST FOR PUBLICATION)
    model_hier, hier_comparison = hierarchical_regression(df)
    
    # Solution 2: Separate Models
    separate_results = separate_models(df)
    
    # Solution 3: Mediation Analysis
    mediation_results = mediation_analysis(df)
    
    # Solution 4: Principal Components Regression
    model_pca, loadings = principal_components_regression(df)
    
    # Solution 5: Ridge Regression
    ridge_model, ridge_coefs = ridge_regression(df)
    
    # Solution 6: Simplified Model
    simple_model, vif_check = simplified_model(df)
    
    # Visualizations
    visualize_comparisons(hier_comparison, separate_results, mediation_results)
    
    print("\n" + "="*70)
    print("RECOMMENDATIONS FOR YOUR MANUSCRIPT")
    print("="*70)
    
    print("""
    RECOMMENDED APPROACH: Use HIERARCHICAL REGRESSION (Solution 1)
    
    WHY?
    1. Shows how variables contribute sequentially
    2. Demonstrates that Career Autonomy adds significant variance (ΔR²)
    3. Avoids multicollinearity by not including correlated predictors together
    4. Easy to interpret for readers
    5. Standard approach in educational psychology journals
    
    FOR YOUR MANUSCRIPT:
    - Report Model 4 (without Parental Influence) as your main model
    - Mention that Parental Influence was dropped due to high correlation with SES (r > .80)
    - Show hierarchical table demonstrating incremental R²
    - Note that SES and Parental Influence represent overlapping constructs
    
    ALTERNATIVE APPROACH: Report SEPARATE MODELS (Solution 2)
    - Show that when examined independently, each predictor has expected relationships
    - Useful for supplementary analyses
    
    ADVANCED APPROACH: Report MEDIATION ANALYSIS (Solution 3)
    - If mediation is supported, this elegantly explains why multicollinearity exists
    - Shows the causal pathway: SES → Parental Influence → Career Satisfaction
    """)
    
    # Save all results
    hier_comparison.to_csv('hierarchical_regression_comparison.csv', index=False)
    separate_results.to_csv('separate_models_results.csv', index=False)
    ridge_coefs.to_csv('ridge_regression_coefficients.csv', index=False)
    
    print("\n✓ All results saved!")
    print("  - hierarchical_regression_comparison.csv")
    print("  - separate_models_results.csv")
    print("  - ridge_regression_coefficients.csv")
    print("  - 09_multicollinearity_solutions.png")
    print("  - 10_mediation_diagram.png")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()