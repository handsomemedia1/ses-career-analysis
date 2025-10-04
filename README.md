# Career Autonomy Over Advantage: SES and Career Satisfaction in Nigeria

**A quantitative analysis of how socioeconomic status and personal agency shape career satisfaction among Nigerian secondary school students**

---

## Project Overview

This repository contains analysis code and visualizations for a cross-sectional study examining career choice determinants among 100 secondary school students from Federal College of Education, Oyo, Nigeria (conducted May 2023).

### Research Question

How do parental socioeconomic status (SES), gender influences, and career autonomy predict career satisfaction among Nigerian adolescents?

### Key Finding

**Career autonomy (choosing careers based on personal interests rather than external pressures) predicts satisfaction more strongly than family socioeconomic resources.** This finding challenges deficit-focused narratives about low-SES students and suggests interventions should foster self-determination, not just provide resources.

---

## Key Research Findings

### Main Results

1. **Career Autonomy is the Strongest Predictor**
   - Career autonomy: β = .24, p = .024 (significant)
   - SES composite: β = .17, p = .103 (trending but not significant)
   - Model explains 12% of variance in career satisfaction

2. **Parental Influence Paradox**
   - Despite high correlation with SES (r = .85), parental influence behaviors (encouragement, information provision, financial support) showed NO independent relationship with career satisfaction (β = .05, p = .620)
   - Suggests objective resources matter more than parenting behaviors in resource-constrained contexts

3. **Three Distinct Student Profiles Identified**
   - **"Constrained"** (22%): Low SES, low autonomy, high gender constraints
   - **"Autonomous"** (40%): Low SES but HIGH autonomy and satisfaction
   - **"Advantaged"** (38%): High SES and high autonomy
   
   The "Autonomous" group demonstrates that personal agency can buffer socioeconomic disadvantages.

4. **No Gender Differences**
   - Males and females showed no significant differences in career satisfaction (d = -0.13), career autonomy (d = -0.28), or SES (d = 0.13)

### Theoretical Implications

- Supports Self-Determination Theory: Autonomy is a universal psychological need
- Extends Social Cognitive Career Theory to Nigerian context
- Challenges assumption that parental involvement uniformly benefits career development
- Suggests interventions should foster autonomy-supportive environments, not just provide resources

---

## Repository Structure

```
ses-career-analysis/
│
├── advanced_analysis.py                    # Main statistical analysis script
├── multicollinearity_solutions.py          # Advanced regression solutions
│
├── 01_demographics.png                     # Demographics visualization
├── 02_correlation_heatmap.png              # Correlation matrix
├── 03_ses_career_satisfaction.png          # SES-satisfaction relationships
├── 04_parental_influence.png               # Parental influence components
├── 05_career_choice_basis.png              # What drives career choice
├── 06_cluster_visualization.png            # Three student profiles (PCA)
├── 07_gender_comparisons.png               # Gender differences (none found)
├── 08_composite_distributions.png          # Score distributions
├── 09_multicollinearity_solutions.png      # Regression model comparisons
├── 10_mediation_diagram.png                # Mediation analysis results
│
├── .gitignore                              # Excludes data files
├── README.md                               # This file
└── requirements.txt                        # Python dependencies
```

**Note:** Raw data files (`*.csv`) are excluded from this repository to protect participant confidentiality per research ethics requirements.

---

## Installation & Setup

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Install Required Packages

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn statsmodels
```

### Required Packages

- **pandas** (1.5.0+): Data manipulation
- **numpy** (1.21.0+): Numerical computing
- **matplotlib** (3.5.0+): Visualization
- **seaborn** (0.12.0+): Statistical visualization
- **scipy** (1.9.0+): Statistical tests
- **scikit-learn** (1.1.0+): Machine learning (clustering, PCA)
- **statsmodels** (0.13.0+): Regression analysis

---

## Usage

### Running the Analysis

The analysis pipeline consists of two scripts that should be run sequentially:

#### 1. Main Statistical Analysis

```bash
python advanced_analysis.py
```

**What it does:**
- Loads and cleans data
- Creates composite scores (SES, Parental Influence, Gender Influence, Career Autonomy)
- Performs descriptive statistics
- Correlation analysis with 95% confidence intervals
- Group comparisons (t-tests, ANOVA) with effect sizes
- Multiple regression analysis
- K-means cluster analysis (3 profiles)
- Factor analysis / PCA
- Generates 8 publication-quality visualizations

**Output:**
- Console output with all statistical results
- 8 PNG files (01-08)
- `analyzed_data_with_composites.csv` (processed data)

#### 2. Multicollinearity Solutions

```bash
python multicollinearity_solutions.py
```

**What it does:**
- Addresses high correlation between SES and Parental Influence (r = .85)
- Hierarchical regression (sequential variable entry)
- Separate models for each predictor
- Mediation analysis (tests if Parental Influence mediates SES effects)
- Principal components regression
- Ridge regression
- Simplified model (drops collinear predictors)

**Output:**
- Console output comparing 6 different approaches
- 2 PNG files (09-10)
- 3 CSV files with detailed results

### Expected Runtime

- `advanced_analysis.py`: 10-15 seconds
- `multicollinearity_solutions.py`: 5-10 seconds

---

## Understanding the Output

### Console Output Sections

Both scripts provide structured console output:

1. **Data Loading**: Sample size, missing data report
2. **Descriptive Statistics**: Demographics, SES distribution, means/SDs
3. **Correlation Analysis**: Bivariate relationships with CIs and effect size interpretations
4. **Group Comparisons**: Gender differences, SES category differences with Cohen's d and eta-squared
5. **Regression Results**: Full OLS output with VIF diagnostics
6. **Cluster Profiles**: Characteristics of three student groups
7. **Factor Loadings**: Underlying dimensions

### Key Statistical Outputs

**Effect Size Interpretations:**
- Correlation (r): small = .10, medium = .30, large = .50
- Cohen's d: small = .20, medium = .50, large = .80
- Eta-squared (η²): small = .01, medium = .06, large = .14

**Significance Levels:**
- p < .05: statistically significant
- p < .01: highly significant
- p < .001: very highly significant

### Visualization Guide

| File | Description | Key Insight |
|------|-------------|-------------|
| 01_demographics.png | Sample characteristics | 60% female, 50% medium SES, 65% received career guidance |
| 02_correlation_heatmap.png | Variable relationships | Career autonomy most strongly correlates with satisfaction |
| 03_ses_career_satisfaction.png | SES-satisfaction link | Positive but modest relationship (r = .22) |
| 04_parental_influence.png | Parental behavior distributions | Most parents provide moderate support |
| 05_career_choice_basis.png | What drives choices | Interests > Academics > Parents |
| 06_cluster_visualization.png | Three student profiles | Shows "Autonomous" group with low SES but high satisfaction |
| 07_gender_comparisons.png | Male vs. female | No significant differences |
| 08_composite_distributions.png | Score distributions | Shows variability in autonomy and SES |
| 09_multicollinearity_solutions.png | Regression comparisons | Career autonomy adds unique variance |
| 10_mediation_diagram.png | Mediation pathway | Parental influence does NOT mediate SES effects |

---

## Methodology Details

### Sample

- **N = 100** secondary school students
- **Location:** Federal College of Education (Special), Oyo, Nigeria
- **Data collection:** May 2023
- **Age distribution:** 13% youngest, 37% age 2, 36% age 3, 14% oldest
- **Gender:** 60% female, 40% male
- **Grade levels:** 51% Level 4 (final year), 25% Level 2, 19% Level 3, 5% Level 1

### Measures

**Socioeconomic Status (SES):** 
- 3 items: parental education, income, social status (4-point scale)
- Composite: M = 2.58, SD = 0.59, α = .78
- Categorized: Low (34%), Medium (50%), High (16%)

**Parental Influence:** 
- 3 items: encouragement, information provision, financial support
- M = 2.97, SD = 0.58, α = .71

**Gender Influence:** 
- 4 items: gender's effect on choice, challenges, limitations, societal roles
- M = 2.04, SD = 0.58, α = .65

**Career Autonomy:** 
- Difference score: interest-based choice minus parent-directed choice
- M = 1.27, SD = 1.32
- Positive = autonomous, Negative = parent-directed

**Career Satisfaction:** 
- Single item: "I am satisfied with my current career choice" (1-4 scale)
- M = 3.14, SD = 0.84

### Statistical Approach

**Addressing Multicollinearity:**
- SES and Parental Influence highly correlated (r = .85, VIF > 20)
- Solution: Hierarchical regression with sequential entry
- Parental Influence excluded from final model (no independent effect: β = .05, p = .620)

**Final Model:**
```
Career Satisfaction = β₀ + β₁(Gender) + β₂(SES) + β₃(Gender Influence) + β₄(Career Autonomy)

Results: R² = .120, F(4,95) = 3.24, p = .016
Only Career Autonomy significant: β = .24, p = .024
```

**Robustness Checks:**
- Separate models for each predictor
- Mediation analysis
- Cluster analysis for heterogeneity
- VIF diagnostics

---

## Interpretation Guide

### For Researchers

**Replicating this analysis:**
1. Prepare CSV with same variable structure (see Methodology section)
2. Ensure column names match those in `advanced_analysis.py` (or modify script)
3. Run scripts in order
4. Interpret effect sizes, not just p-values
5. Address multicollinearity if present (VIF > 10)

**Adapting for your context:**
- Modify composite score calculations for your measures
- Adjust cluster number (currently k=3) based on your sample
- Add control variables as needed
- Modify visualization aesthetics

### For Practitioners

**Key takeaways for career guidance:**

1. **Foster Autonomy First**: Students who choose careers based on personal interests are more satisfied, regardless of family wealth

2. **Don't Assume Resources Equal Success**: The "Autonomous" cluster (40% of sample) maintained high satisfaction despite low SES

3. **Parental Involvement ≠ Career Satisfaction**: Quantity of parental support doesn't predict satisfaction; quality and autonomy-support may matter more

4. **Gender Equity is Possible**: This Nigerian sample showed no gender differences, suggesting appropriate interventions can achieve equity

**Intervention Implications:**
- Implement autonomy-supportive career counseling
- Teach decision-making skills, not just provide information
- Help students clarify personal values and interests
- Acknowledge external pressures while empowering personal choice

---

## Ethical Considerations

### Data Protection

**Why raw data is excluded:**
- Participant responses may contain identifying information
- Research ethics requires confidentiality protection
- Informed consent did not include public data sharing

**What is shared:**
- Analysis code (fully reproducible)
- Aggregate results and visualizations
- No individual-level data

### Reproducibility Without Raw Data

Researchers wishing to replicate this analysis can:
1. Use the provided code with their own data
2. Request de-identified data from authors (if IRB permits)
3. Contact authors for collaboration on similar studies

---

## Citation

If you use this code or build upon this research, please cite:

```
[Author names]. (2025). Career autonomy over advantage: How socioeconomic 
status and personal agency shape career satisfaction among Nigerian secondary 
students. [Journal Name], [Volume(Issue)], [pages]. 
https://doi.org/[DOI when available]
```

**Preprint:** [Link when available]

---

## Contributing

This is a completed research project, but we welcome:
- Bug reports or code improvements (open an issue)
- Replications in other contexts (contact authors)
- Extensions or follow-up studies (collaboration inquiries welcome)

---

## License

**Code:** MIT License (see LICENSE file)
- You may use, modify, and distribute the analysis code
- Attribution required

**Research Findings:** All rights reserved
- Manuscript content and findings remain under standard academic copyright
- Cite appropriately if discussing or building upon results

---

## Contact & Collaboration

**Primary Investigator:** [Your name]  
**Institution:** [Your institution]  
**Email:** [Your email]

**For inquiries about:**
- Data access requests
- Collaboration opportunities
- Methodological questions
- Replication in other countries

**Research Team:** [List all co-authors/supervisors]

---

## Acknowledgments

We thank:
- Students and staff at Federal College of Education (Special), Oyo, Nigeria
- [Funding sources, if any]
- [Advisors/supervisors]
- [Any other contributors]

---

## Related Publications & Resources

### From This Project
- [Link to published paper when available]
- [Link to conference presentations]
- [Link to preprint server]

### Related Work
- [Your other career development research]
- [Supervisor's related publications]

### Relevant Datasets
- World Bank Nigeria Education Statistics
- ILO Global Employment Trends for Youth 2024
- Afrobarometer Round 10 (Nigeria)

---

## Frequently Asked Questions

**Q: Can I access the raw data?**  
A: Raw data contains participant information and cannot be shared publicly. Contact the authors to discuss potential collaboration or data sharing under appropriate agreements.

**Q: How do I adapt this code for my data?**  
A: Ensure your CSV has similar structure (demographics, SES measures, outcome variables). Modify variable names in the scripts to match your column headers.

**Q: What if my SES and parental influence aren't correlated?**  
A: Then multicollinearity won't be an issue. You can include both in your regression model. The hierarchical approach is still useful for showing incremental variance.

**Q: Why are some p-values borderline (.103, .072)?**  
A: With N=100, power is limited for detecting small effects. We emphasize effect sizes and confidence intervals alongside p-values. Larger samples would increase precision.

**Q: Can I use this code for longitudinal data?**  
A: The current code is for cross-sectional analysis. Longitudinal data requires additional methods (mixed models, growth curves, panel regression). Contact us for longitudinal adaptation.

**Q: What journals should I target for similar research?**  
A: High-impact options include *Journal of Vocational Behavior*, *International Journal for Educational and Vocational Guidance*, *Career Development International*, and regional journals like *African Journal of Career Development*.

---

## Version History

**v1.0.0** (October 2025)
- Initial public release
- Complete analysis pipeline
- 10 publication-quality visualizations
- Comprehensive documentation

---

## Technical Notes

**Tested Environment:**
- Python 3.13
- Windows 11
- All dependencies current as of October 2025

**Known Issues:**
- None currently identified

**Future Enhancements:**
- Interactive visualizations (Plotly)
- Shiny/Dash dashboard
- Longitudinal analysis module
- Cross-country comparison tools

---

**Last Updated:** October 4, 2025  
**Status:** Active Research Project  
**Manuscript Status:** Under preparation for submission

---

*This research was conducted in accordance with institutional ethical guidelines and received approval from the relevant ethics review board.*