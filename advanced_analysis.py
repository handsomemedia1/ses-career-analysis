"""
Advanced Statistical Analysis and Visualization
Parental Socio-Economic Status and Career Choice Among Nigerian Secondary Students
Author: Research Team
Date: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency, spearmanr, pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LinearRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.formula.api import ols
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality plot style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

# ============================================================================
# 1. DATA LOADING AND PREPROCESSING
# ============================================================================

def load_and_clean_data(filepath):
    """Load CSV and prepare data for analysis"""
    print("="*70)
    print("LOADING AND CLEANING DATA")
    print("="*70)
    
    # Read CSV
    df = pd.read_csv(filepath)
    
    # Remove empty columns and calculated summary columns
    cols_to_drop = [col for col in df.columns if col.startswith('_') or 
                    col == '' or 
                    'Total of' in col or 
                    col == 'Demacation of  Variables']
    df = df.drop(columns=cols_to_drop, errors='ignore')
    
    # Rename columns for easier handling
    column_mapping = {
        'Have you ever received any career Guidance?': 'Career_Guidance_Received',
        'My parents have a high level of education.': 'Parent_Education',
        'My parents have a high-income level.': 'Parent_Income',
        'My parents have a high social status.': 'Parent_Social_Status',
        'My parents actively encouraged me to pursue a certain career path.': 'Parent_Encouragement',
        'My gender influences my career choice.': 'Gender_Influences_Choice',
        'I have faced challenges in choosing a career based on my gender.': 'Gender_Challenges',
        'My gender does not limit my career choices.': 'No_Gender_Limits',
        'Gender roles in society have influenced my career choice.': 'Gender_Roles_Influence',
        'My parents have discouraged me from pursuing a certain career path.': 'Parent_Discouragement',
        'My parents have provided me with information about different career options.': 'Parent_Info_Provision',
        'My parents have provided me with financial support for my education.': 'Parent_Financial_Support',
        'I chose my career path based on my interests and passions.': 'Choice_Based_Interests',
        'I chose my career path based on my parents\' suggestions or expectations.': 'Choice_Based_Parents',
        'I chose my career path based on my academic strengths and weaknesses.': 'Choice_Based_Academics',
        'I am satisfied with my current career choice.': 'Career_Satisfaction',
        'I would choose a different career path if given the opportunity.': 'Would_Change_Career'
    }
    
    df = df.rename(columns=column_mapping)
    
    # Convert timestamp to datetime
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    
    # Remove rows with excessive missing data (keep only substantive responses)
    df = df.dropna(thresh=10)
    
    print(f"\nDataset shape: {df.shape}")
    print(f"Number of participants: {len(df)}")
    print(f"\nMissing values per column:")
    print(df.isnull().sum()[df.isnull().sum() > 0])
    
    return df

def create_composite_scores(df):
    """Create composite SES and other indices"""
    print("\n" + "="*70)
    print("CREATING COMPOSITE SCORES")
    print("="*70)
    
    # Create SES Composite (average of education, income, social status)
    df['SES_Composite'] = df[['Parent_Education', 'Parent_Income', 'Parent_Social_Status']].mean(axis=1)
    
    # Create Parental Influence Score
    parental_cols = ['Parent_Encouragement', 'Parent_Info_Provision', 'Parent_Financial_Support']
    df['Parental_Influence_Score'] = df[parental_cols].mean(axis=1)
    
    # Create Gender Influence Score (reverse code "No_Gender_Limits")
    df['No_Gender_Limits_Rev'] = 5 - df['No_Gender_Limits']  # Reverse coding
    gender_cols = ['Gender_Influences_Choice', 'Gender_Challenges', 'No_Gender_Limits_Rev', 'Gender_Roles_Influence']
    df['Gender_Influence_Score'] = df[gender_cols].mean(axis=1)
    
    # Create Career Decision Autonomy Score
    df['Career_Autonomy'] = df['Choice_Based_Interests'] - df['Choice_Based_Parents']
    
    # Categorize SES into Low, Medium, High
    df['SES_Category'] = pd.cut(df['SES_Composite'], bins=3, labels=['Low SES', 'Medium SES', 'High SES'])
    
    # Recode Gender (assuming 1=Male, 2=Female)
    df['Gender_Label'] = df['Gender'].map({1: 'Male', 2: 'Female'})
    
    print("\nComposite scores created:")
    print("- SES_Composite (range:", df['SES_Composite'].min(), "-", df['SES_Composite'].max(), ")")
    print("- Parental_Influence_Score")
    print("- Gender_Influence_Score")
    print("- Career_Autonomy")
    
    return df

# ============================================================================
# 2. DESCRIPTIVE STATISTICS WITH EFFECT SIZES
# ============================================================================

def descriptive_statistics(df):
    """Comprehensive descriptive statistics"""
    print("\n" + "="*70)
    print("DESCRIPTIVE STATISTICS")
    print("="*70)
    
    # Demographics
    print("\n--- DEMOGRAPHICS ---")
    print(f"\nGender Distribution:")
    print(df['Gender_Label'].value_counts())
    print(f"\nAge Distribution:")
    print(df['Age'].value_counts().sort_index())
    print(f"\nLevel of Study:")
    print(df['Level Of Study'].value_counts().sort_index())
    print(f"\nCareer Guidance Received:")
    print(df['Career_Guidance_Received'].value_counts())
    
    # SES Distribution
    print("\n--- SES DISTRIBUTION ---")
    print(df['SES_Category'].value_counts())
    print(f"\nSES Composite Statistics:")
    print(df['SES_Composite'].describe())
    
    # Career Satisfaction
    print("\n--- CAREER SATISFACTION ---")
    print(f"Mean Career Satisfaction: {df['Career_Satisfaction'].mean():.2f} (SD={df['Career_Satisfaction'].std():.2f})")
    print(f"Would Change Career (Mean): {df['Would_Change_Career'].mean():.2f}")
    
    # Composite Scores
    print("\n--- COMPOSITE SCORES ---")
    for col in ['Parental_Influence_Score', 'Gender_Influence_Score', 'Career_Autonomy']:
        print(f"{col}: Mean={df[col].mean():.2f}, SD={df[col].std():.2f}")
    
    return df

# ============================================================================
# 3. CORRELATION ANALYSIS WITH CONFIDENCE INTERVALS
# ============================================================================

def correlation_analysis(df):
    """Correlation analysis with effect sizes and CI"""
    print("\n" + "="*70)
    print("CORRELATION ANALYSIS")
    print("="*70)
    
    # Select key variables
    cor_vars = ['SES_Composite', 'Parental_Influence_Score', 'Gender_Influence_Score',
                'Career_Autonomy', 'Career_Satisfaction', 'Choice_Based_Interests',
                'Choice_Based_Parents', 'Choice_Based_Academics']
    
    cor_data = df[cor_vars].dropna()
    
    # Calculate correlations
    cor_matrix = cor_data.corr()
    
    print("\nKey Correlations with Career Satisfaction:")
    print("-" * 50)
    for var in ['SES_Composite', 'Parental_Influence_Score', 'Gender_Influence_Score', 'Career_Autonomy']:
        r, p = pearsonr(cor_data[var], cor_data['Career_Satisfaction'])
        
        # Calculate 95% CI for correlation using Fisher's z-transformation
        n = len(cor_data)
        z = np.arctanh(r)
        se = 1/np.sqrt(n-3)
        z_crit = 1.96
        ci_lower = np.tanh(z - z_crit*se)
        ci_upper = np.tanh(z + z_crit*se)
        
        # Effect size interpretation
        if abs(r) < 0.1:
            effect = "negligible"
        elif abs(r) < 0.3:
            effect = "small"
        elif abs(r) < 0.5:
            effect = "medium"
        else:
            effect = "large"
        
        print(f"{var}:")
        print(f"  r = {r:.3f}, p = {p:.4f}, 95% CI [{ci_lower:.3f}, {ci_upper:.3f}]")
        print(f"  Effect size: {effect}, R² = {r**2:.3f}")
    
    return cor_matrix

# ============================================================================
# 4. GROUP COMPARISONS WITH EFFECT SIZES
# ============================================================================

def group_comparisons(df):
    """Compare groups with effect sizes (Cohen's d)"""
    print("\n" + "="*70)
    print("GROUP COMPARISONS")
    print("="*70)
    
    def cohens_d(group1, group2):
        """Calculate Cohen's d effect size"""
        n1, n2 = len(group1), len(group2)
        var1, var2 = group1.var(), group2.var()
        pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
        return (group1.mean() - group2.mean()) / pooled_std
    
    # Gender differences
    print("\n--- GENDER DIFFERENCES ---")
    males = df[df['Gender_Label'] == 'Male']
    females = df[df['Gender_Label'] == 'Female']
    
    for var in ['Career_Satisfaction', 'Career_Autonomy', 'SES_Composite']:
        t_stat, p_val = stats.ttest_ind(males[var].dropna(), females[var].dropna())
        d = cohens_d(males[var].dropna(), females[var].dropna())
        
        print(f"\n{var}:")
        print(f"  Male: M={males[var].mean():.2f}, SD={males[var].std():.2f}")
        print(f"  Female: M={females[var].mean():.2f}, SD={females[var].std():.2f}")
        print(f"  t={t_stat:.3f}, p={p_val:.4f}, Cohen's d={d:.3f}")
    
    # SES Category differences in career satisfaction
    print("\n--- SES CATEGORY DIFFERENCES ---")
    ses_groups = [df[df['SES_Category'] == cat]['Career_Satisfaction'].dropna() 
                  for cat in ['Low SES', 'Medium SES', 'High SES']]
    
    f_stat, p_val = stats.f_oneway(*ses_groups)
    print(f"\nCareer Satisfaction by SES:")
    print(f"  F={f_stat:.3f}, p={p_val:.4f}")
    
    for cat in ['Low SES', 'Medium SES', 'High SES']:
        group_data = df[df['SES_Category'] == cat]['Career_Satisfaction']
        print(f"  {cat}: M={group_data.mean():.2f}, SD={group_data.std():.2f}, N={len(group_data)}")
    
    # Calculate eta-squared (effect size for ANOVA)
    ss_between = sum([len(g) * (g.mean() - df['Career_Satisfaction'].mean())**2 for g in ses_groups])
    ss_total = sum([(x - df['Career_Satisfaction'].mean())**2 for g in ses_groups for x in g])
    eta_squared = ss_between / ss_total
    print(f"  η² = {eta_squared:.3f}")

# ============================================================================
# 5. MULTIPLE REGRESSION ANALYSIS
# ============================================================================

def regression_analysis(df):
    """Multiple regression with VIF and model diagnostics"""
    print("\n" + "="*70)
    print("MULTIPLE REGRESSION ANALYSIS")
    print("="*70)
    
    # Prepare data
    reg_data = df[['Career_Satisfaction', 'SES_Composite', 'Parental_Influence_Score',
                   'Gender_Influence_Score', 'Career_Autonomy', 'Gender']].dropna()
    
    # Dependent variable
    y = reg_data['Career_Satisfaction']
    
    # Independent variables
    X = reg_data[['SES_Composite', 'Parental_Influence_Score', 
                  'Gender_Influence_Score', 'Career_Autonomy', 'Gender']]
    
    # Add constant
    X_const = sm.add_constant(X)
    
    # Fit model
    model = sm.OLS(y, X_const).fit()
    
    print("\nRegression Results:")
    print("="*50)
    print(model.summary())
    
    # Calculate VIF for multicollinearity
    print("\n--- MULTICOLLINEARITY CHECK (VIF) ---")
    vif_data = pd.DataFrame()
    vif_data["Variable"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    print(vif_data)
    print("\nNote: VIF > 10 indicates problematic multicollinearity")
    
    # Effect sizes (standardized coefficients)
    print("\n--- STANDARDIZED COEFFICIENTS (Beta) ---")
    X_std = StandardScaler().fit_transform(X)
    y_std = StandardScaler().fit_transform(y.values.reshape(-1, 1)).flatten()
    X_std_const = sm.add_constant(X_std)
    model_std = sm.OLS(y_std, X_std_const).fit()
    
    for i, var in enumerate(X.columns):
        print(f"{var}: β = {model_std.params[i+1]:.3f}, p = {model_std.pvalues[i+1]:.4f}")
    
    return model

# ============================================================================
# 6. CLUSTER ANALYSIS
# ============================================================================

def cluster_analysis(df):
    """K-means and hierarchical clustering"""
    print("\n" + "="*70)
    print("CLUSTER ANALYSIS")
    print("="*70)
    
    # Prepare data for clustering
    cluster_vars = ['SES_Composite', 'Parental_Influence_Score', 
                    'Gender_Influence_Score', 'Career_Autonomy']
    cluster_data = df[cluster_vars].dropna()
    
    # Standardize
    scaler = StandardScaler()
    cluster_data_scaled = scaler.fit_transform(cluster_data)
    
    # Determine optimal number of clusters using elbow method
    inertias = []
    silhouettes = []
    K_range = range(2, 7)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(cluster_data_scaled)
        inertias.append(kmeans.inertia_)
        silhouettes.append(silhouette_score(cluster_data_scaled, kmeans.labels_))
    
    # Use 3 clusters (typical for Low/Medium/High groups)
    optimal_k = 3
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(cluster_data_scaled)
    
    print(f"\nK-Means Clustering with k={optimal_k}")
    print(f"Silhouette Score: {silhouette_score(cluster_data_scaled, cluster_labels):.3f}")
    
    # Add cluster labels to dataframe
    cluster_data['Cluster'] = cluster_labels
    
    # Describe clusters
    print("\n--- CLUSTER PROFILES ---")
    for i in range(optimal_k):
        print(f"\nCluster {i} (n={sum(cluster_labels==i)}):")
        cluster_subset = cluster_data[cluster_data['Cluster'] == i][cluster_vars]
        print(cluster_subset.mean())
    
    return cluster_labels, cluster_data

# ============================================================================
# 7. FACTOR ANALYSIS / PCA
# ============================================================================

def factor_analysis(df):
    """Exploratory Factor Analysis"""
    print("\n" + "="*70)
    print("FACTOR ANALYSIS")
    print("="*70)
    
    # Select all Likert-scale items
    factor_vars = ['Parent_Education', 'Parent_Income', 'Parent_Social_Status',
                   'Parent_Encouragement', 'Parent_Info_Provision', 'Parent_Financial_Support',
                   'Gender_Influences_Choice', 'Gender_Challenges', 'Gender_Roles_Influence',
                   'Choice_Based_Interests', 'Choice_Based_Parents', 'Choice_Based_Academics',
                   'Career_Satisfaction']
    
    factor_data = df[factor_vars].dropna()
    
    # Standardize
    scaler = StandardScaler()
    factor_data_scaled = scaler.fit_transform(factor_data)
    
    # PCA
    pca = PCA()
    pca.fit(factor_data_scaled)
    
    print("\n--- PRINCIPAL COMPONENTS ANALYSIS ---")
    print(f"Explained variance ratio (first 5 components):")
    for i, var in enumerate(pca.explained_variance_ratio_[:5]):
        print(f"  PC{i+1}: {var:.3f} ({var*100:.1f}%)")
    
    print(f"\nCumulative explained variance (first 5 components): {pca.explained_variance_ratio_[:5].sum():.3f}")
    
    # Factor Analysis with 3 factors
    n_factors = 3
    fa = FactorAnalysis(n_components=n_factors, random_state=42)
    fa.fit(factor_data_scaled)
    
    print(f"\n--- FACTOR LOADINGS (3 factors) ---")
    loadings = pd.DataFrame(
        fa.components_.T,
        columns=[f'Factor {i+1}' for i in range(n_factors)],
        index=factor_vars
    )
    print(loadings.round(3))

# ============================================================================
# 8. ADVANCED VISUALIZATIONS
# ============================================================================

def create_visualizations(df, cor_matrix, cluster_labels, cluster_data):
    """Create publication-quality visualizations"""
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    
    # 1. Demographics Overview
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Gender distribution
    gender_counts = df['Gender_Label'].value_counts()
    axes[0, 0].bar(gender_counts.index, gender_counts.values, color=['#3498db', '#e74c3c'])
    axes[0, 0].set_title('Gender Distribution', fontweight='bold')
    axes[0, 0].set_ylabel('Count')
    for i, v in enumerate(gender_counts.values):
        axes[0, 0].text(i, v + 1, str(v), ha='center', fontweight='bold')
    
    # SES Category distribution
    ses_counts = df['SES_Category'].value_counts()
    axes[0, 1].bar(range(len(ses_counts)), ses_counts.values, 
                   color=['#e74c3c', '#f39c12', '#2ecc71'])
    axes[0, 1].set_title('SES Category Distribution', fontweight='bold')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_xticks(range(len(ses_counts)))
    axes[0, 1].set_xticklabels(ses_counts.index, rotation=0)
    for i, v in enumerate(ses_counts.values):
        axes[0, 1].text(i, v + 1, str(v), ha='center', fontweight='bold')
    
    # Career Guidance Received
    guidance_counts = df['Career_Guidance_Received'].value_counts()
    axes[1, 0].bar(['No (2)', 'Yes (1)'], [guidance_counts.get(2, 0), guidance_counts.get(1, 0)],
                   color=['#e74c3c', '#2ecc71'])
    axes[1, 0].set_title('Career Guidance Received', fontweight='bold')
    axes[1, 0].set_ylabel('Count')
    for i, label in enumerate(['No (2)', 'Yes (1)']):
        val = guidance_counts.get(2-i, 0)
        axes[1, 0].text(i, val + 1, str(val), ha='center', fontweight='bold')
    
    # Age distribution
    age_counts = df['Age'].value_counts().sort_index()
    axes[1, 1].bar(age_counts.index, age_counts.values, color='#9b59b6')
    axes[1, 1].set_title('Age Distribution', fontweight='bold')
    axes[1, 1].set_xlabel('Age Code')
    axes[1, 1].set_ylabel('Count')
    
    plt.tight_layout()
    plt.savefig('01_demographics.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: 01_demographics.png")
    plt.close()
    
    # 2. Correlation Heatmap
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(cor_matrix, dtype=bool))
    sns.heatmap(cor_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, vmin=-1, vmax=1, square=True, linewidths=1,
                cbar_kws={"shrink": 0.8})
    plt.title('Correlation Matrix: Key Variables', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('02_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: 02_correlation_heatmap.png")
    plt.close()
    
    # 3. SES and Career Satisfaction
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Box plot
    ses_categories = ['Low SES', 'Medium SES', 'High SES']
    data_for_box = [df[df['SES_Category'] == cat]['Career_Satisfaction'].dropna() 
                    for cat in ses_categories]
    bp = axes[0].boxplot(data_for_box, labels=ses_categories, patch_artist=True)
    for patch, color in zip(bp['boxes'], ['#e74c3c', '#f39c12', '#2ecc71']):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[0].set_title('Career Satisfaction by SES Category', fontweight='bold')
    axes[0].set_ylabel('Career Satisfaction Score')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Scatter plot with regression line
    valid_data = df[['SES_Composite', 'Career_Satisfaction']].dropna()
    axes[1].scatter(valid_data['SES_Composite'], valid_data['Career_Satisfaction'], 
                    alpha=0.6, s=80, color='#3498db')
    
    # Add regression line
    z = np.polyfit(valid_data['SES_Composite'], valid_data['Career_Satisfaction'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(valid_data['SES_Composite'].min(), 
                         valid_data['SES_Composite'].max(), 100)
    axes[1].plot(x_line, p(x_line), "r--", linewidth=2, label='Regression Line')
    
    # Add correlation
    r, p_val = pearsonr(valid_data['SES_Composite'], valid_data['Career_Satisfaction'])
    axes[1].text(0.05, 0.95, f'r = {r:.3f}\np = {p_val:.4f}', 
                 transform=axes[1].transAxes, fontsize=11,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    axes[1].set_title('SES vs Career Satisfaction', fontweight='bold')
    axes[1].set_xlabel('SES Composite Score')
    axes[1].set_ylabel('Career Satisfaction')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('03_ses_career_satisfaction.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: 03_ses_career_satisfaction.png")
    plt.close()
    
    # 4. Parental Influence Analysis
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Distribution of parental influence components
    parental_vars = ['Parent_Encouragement', 'Parent_Info_Provision', 
                     'Parent_Financial_Support', 'Parent_Discouragement']
    for i, var in enumerate(parental_vars):
        ax = axes[i//2, i%2]
        data = df[var].value_counts().sort_index()
        ax.bar(data.index, data.values, color='#16a085', alpha=0.7)
        ax.set_title(var.replace('_', ' '), fontweight='bold')
        ax.set_xlabel('Response (1-4 scale)')
        ax.set_ylabel('Count')
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('04_parental_influence.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: 04_parental_influence.png")
    plt.close()
    
    # 5. Career Choice Basis
    fig, ax = plt.subplots(figsize=(10, 6))
    choice_vars = ['Choice_Based_Interests', 'Choice_Based_Parents', 'Choice_Based_Academics']
    choice_means = [df[var].mean() for var in choice_vars]
    choice_std = [df[var].std() for var in choice_vars]
    
    x_pos = np.arange(len(choice_vars))
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    bars = ax.bar(x_pos, choice_means, yerr=choice_std, capsize=5, 
                  color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    ax.set_xlabel('Basis of Career Choice', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Score', fontsize=12, fontweight='bold')
    ax.set_title('What Drives Career Choice?', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['Interests', 'Parents', 'Academics'], fontsize=11)
    ax.set_ylim(0, 5)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, mean) in enumerate(zip(bars, choice_means)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{mean:.2f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('05_career_choice_basis.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: 05_career_choice_basis.png")
    plt.close()
    
    # 6. Cluster Visualization (PCA)
    pca = PCA(n_components=2)
    cluster_vars = ['SES_Composite', 'Parental_Influence_Score', 
                    'Gender_Influence_Score', 'Career_Autonomy']
    cluster_data_viz = cluster_data[cluster_vars].values
    cluster_pca = pca.fit_transform(StandardScaler().fit_transform(cluster_data_viz))
    
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(cluster_pca[:, 0], cluster_pca[:, 1], 
                        c=cluster_labels, cmap='viridis', s=100, alpha=0.6, edgecolors='black')
    
    # Add cluster centers
    scaler = StandardScaler()
    cluster_data_scaled = scaler.fit_transform(cluster_data_viz)
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans.fit(cluster_data_scaled)
    centers_pca = pca.transform(kmeans.cluster_centers_)
    ax.scatter(centers_pca[:, 0], centers_pca[:, 1], 
              c='red', s=300, alpha=1, edgecolors='black', linewidths=2, 
              marker='*', label='Cluster Centers')
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)', 
                  fontsize=12, fontweight='bold')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)', 
                  fontsize=12, fontweight='bold')
    ax.set_title('Student Clusters Based on SES and Career Factors', 
                fontsize=14, fontweight='bold', pad=15)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.colorbar(scatter, label='Cluster')
    
    plt.tight_layout()
    plt.savefig('06_cluster_visualization.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: 06_cluster_visualization.png")
    plt.close()
    
    # 7. Gender Comparisons
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    gender_vars = ['Career_Satisfaction', 'Career_Autonomy', 'SES_Composite']
    for i, var in enumerate(gender_vars):
        males = df[df['Gender_Label'] == 'Male'][var].dropna()
        females = df[df['Gender_Label'] == 'Female'][var].dropna()
        
        bp = axes[i].boxplot([males, females], labels=['Male', 'Female'], patch_artist=True)
        for patch, color in zip(bp['boxes'], ['#3498db', '#e74c3c']):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        axes[i].set_title(var.replace('_', ' '), fontweight='bold')
        axes[i].set_ylabel('Score')
        axes[i].grid(axis='y', alpha=0.3)
    
    plt.suptitle('Gender Comparisons', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('07_gender_comparisons.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: 07_gender_comparisons.png")
    plt.close()
    
    # 8. Composite Score Distributions
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    composite_vars = ['SES_Composite', 'Parental_Influence_Score', 
                     'Gender_Influence_Score', 'Career_Autonomy']
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    
    for i, (var, color) in enumerate(zip(composite_vars, colors)):
        ax = axes[i//2, i%2]
        data = df[var].dropna()
        
        # Histogram with KDE
        ax.hist(data, bins=15, alpha=0.7, color=color, edgecolor='black')
        ax.axvline(data.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {data.mean():.2f}')
        ax.axvline(data.median(), color='blue', linestyle='--', linewidth=2, label=f'Median: {data.median():.2f}')
        
        ax.set_title(var.replace('_', ' '), fontweight='bold')
        ax.set_xlabel('Score')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('08_composite_distributions.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: 08_composite_distributions.png")
    plt.close()
    
    print("\n✓ All visualizations saved successfully!")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Execute full analysis pipeline"""
    print("\n" + "="*70)
    print("ADVANCED STATISTICAL ANALYSIS")
    print("Parental SES and Career Choice - Nigerian Secondary Students")
    print("="*70 + "\n")
    
    # Load and prepare data
    df = load_and_clean_data('PSES  Form Responses .csv')
    df = create_composite_scores(df)
    
    # Descriptive statistics
    df = descriptive_statistics(df)
    
    # Correlation analysis
    cor_matrix = correlation_analysis(df)
    
    # Group comparisons
    group_comparisons(df)
    
    # Regression analysis
    model = regression_analysis(df)
    
    # Cluster analysis
    cluster_labels, cluster_data = cluster_analysis(df)
    
    # Factor analysis
    factor_analysis(df)
    
    # Create visualizations
    create_visualizations(df, cor_matrix, cluster_labels, cluster_data)
    
    # Save cleaned dataset with composite scores
    output_file = 'analyzed_data_with_composites.csv'
    df.to_csv(output_file, index=False)
    print(f"\n✓ Cleaned dataset saved: {output_file}")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print("\nGenerated Files:")
    print("  - 8 visualization PNG files")
    print("  - analyzed_data_with_composites.csv")
    print("\nNext Steps:")
    print("  1. Review visualizations for patterns")
    print("  2. Interpret regression results for manuscript")
    print("  3. Use cluster profiles to describe student groups")
    print("  4. Report effect sizes (Cohen's d, η², R²) in Results section")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()