# Comprehensive Guide to Correlation, R, and R-squared

## Table of Contents
1. [Introduction](#1-introduction)
2. [Correlation](#2-correlation)
   2.1 [Types of Correlation](#21-types-of-correlation)
   2.2 [Properties of Correlation](#22-properties-of-correlation)
   2.3 [Interpreting Correlation](#23-interpreting-correlation)
3. [Correlation Coefficient (R)](#3-correlation-coefficient-r)
   3.1 [Pearson's Correlation Coefficient](#31-pearsons-correlation-coefficient)
   3.2 [Spearman's Rank Correlation Coefficient](#32-spearmans-rank-correlation-coefficient)
   3.3 [Kendall's Tau](#33-kendalls-tau)
4. [Coefficient of Determination (R-squared)](#4-coefficient-of-determination-r-squared)
   4.1 [Definition and Calculation](#41-definition-and-calculation)
   4.2 [Interpreting R-squared](#42-interpreting-r-squared)
   4.3 [Adjusted R-squared](#43-adjusted-r-squared)
5. [Relationship Between R and R-squared](#5-relationship-between-r-and-r-squared)
6. [Limitations and Considerations](#6-limitations-and-considerations)
7. [Applications in Various Fields](#7-applications-in-various-fields)
8. [Statistical Tests for Correlation](#8-statistical-tests-for-correlation)
9. [Visualizing Correlation and R-squared](#9-visualizing-correlation-and-r-squared)
10. [Advanced Topics](#10-advanced-topics)
11. [Implementing in Python](#11-implementing-in-python)
12. [Conclusion](#12-conclusion)

## 1. Introduction

Correlation, R (correlation coefficient), and R² (coefficient of determination) are fundamental concepts in statistics and data analysis. They are used to measure and describe the relationship between variables, quantify the strength of these relationships, and assess the goodness of fit in regression models.

## 2. Correlation

Correlation is a statistical measure that expresses the extent to which two variables are linearly related. It provides information about the strength and direction of the relationship between variables.

### 2.1 Types of Correlation

1. Positive Correlation: As one variable increases, the other tends to increase.
2. Negative Correlation: As one variable increases, the other tends to decrease.
3. No Correlation: No linear relationship between the variables.

### 2.2 Properties of Correlation

1. Range: Correlation values range from -1 to +1.
2. Symmetry: The correlation between X and Y is the same as the correlation between Y and X.
3. Scale Invariance: Correlation is not affected by changes in the scale of either variable.
4. No Units: Correlation is a dimensionless quantity.

### 2.3 Interpreting Correlation

- Perfect Positive Correlation: +1
- Strong Positive Correlation: 0.7 to 0.9
- Moderate Positive Correlation: 0.5 to 0.7
- Weak Positive Correlation: 0.3 to 0.5
- No Correlation: -0.3 to +0.3
- Weak Negative Correlation: -0.5 to -0.3
- Moderate Negative Correlation: -0.7 to -0.5
- Strong Negative Correlation: -0.9 to -0.7
- Perfect Negative Correlation: -1

Note: These ranges are general guidelines and may vary depending on the context and field of study.

## 3. Correlation Coefficient (R)

The correlation coefficient, often denoted as R, is a specific measure of correlation. There are several types of correlation coefficients, each suited to different types of data and relationships.

### 3.1 Pearson's Correlation Coefficient

Pearson's correlation coefficient (r) measures the strength and direction of the linear relationship between two continuous variables.

Formula:
r = Σ((x - μ_x)(y - μ_y)) / (σ_x σ_y)

Where:
- x and y are the individual sample points
- μ_x and μ_y are the sample means
- σ_x and σ_y are the sample standard deviations

Properties:
- Assumes a linear relationship between variables
- Sensitive to outliers
- Assumes variables are normally distributed

### 3.2 Spearman's Rank Correlation Coefficient

Spearman's correlation (ρ) assesses how well the relationship between two variables can be described using a monotonic function.

Formula:
ρ = 1 - (6 Σ d_i^2) / (n(n^2 - 1))

Where:
- d_i is the difference between the two ranks of each observation
- n is the number of observations

Properties:
- Does not assume linearity
- Less sensitive to outliers than Pearson's
- Can be used with ordinal data

### 3.3 Kendall's Tau

Kendall's tau (τ) measures the ordinal association between two measured quantities.

Formula:
τ = (number of concordant pairs - number of discordant pairs) / (n(n-1)/2)

Properties:
- Robust and has more intuitive interpretation for some applications
- Often used for small sample sizes and when there are many tied ranks

## 4. Coefficient of Determination (R-squared)

R-squared (R²) is a statistical measure that represents the proportion of the variance in the dependent variable that is predictable from the independent variable(s).

### 4.1 Definition and Calculation

R² is defined as the ratio of the explained variation to the total variation:

R² = 1 - (SSres / SStot)

Where:
- SSres is the sum of squares of residuals
- SStot is the total sum of squares

Alternatively, it can be calculated as the square of the correlation coefficient R in simple linear regression.

### 4.2 Interpreting R-squared

- R² ranges from 0 to 1
- R² = 0 indicates that the model explains none of the variability of the response data around its mean
- R² = 1 indicates that the model explains all the variability of the response data around its mean
- Generally, higher R² values indicate better fit, but this should be interpreted cautiously

### 4.3 Adjusted R-squared

Adjusted R-squared is a modified version of R-squared that adjusts for the number of predictors in a model.

Formula:
Adjusted R² = 1 - [(1 - R²)(n - 1) / (n - k - 1)]

Where:
- n is the sample size
- k is the number of predictors

Properties:
- Always lower than or equal to R²
- Increases only if the new term improves the model more than would be expected by chance
- Can be negative

## 5. Relationship Between R and R-squared

In simple linear regression:
- R² is the square of the correlation coefficient R
- R² is always positive, while R can be positive or negative
- The sign of R indicates the direction of the relationship

In multiple regression:
- R² is not simply the square of a single correlation coefficient
- It represents the combined effect of all predictors on the dependent variable

## 6. Limitations and Considerations

1. Correlation does not imply causation
2. Non-linear relationships may not be captured by these measures
3. Outliers can significantly affect correlation and R²
4. R² can increase with the addition of variables, even if they're not meaningful
5. These measures don't account for the quality or reliability of the data

## 7. Applications in Various Fields

1. Economics: Studying relationships between economic indicators
2. Psychology: Analyzing correlations between personality traits
3. Medicine: Investigating relationships between risk factors and diseases
4. Finance: Assessing correlations between different financial instruments
5. Environmental Science: Studying relationships between environmental factors

## 8. Statistical Tests for Correlation

1. Testing significance of Pearson's correlation:
   t = r √((n-2)/(1-r²))
   Where n is the sample size, and t follows a t-distribution with n-2 degrees of freedom

2. Fisher's z-transformation: Used to test differences between correlation coefficients

3. Bootstrapping: Non-parametric approach to estimate confidence intervals for correlation coefficients

## 9. Visualizing Correlation and R-squared

1. Scatter plots: Visualize the relationship between two variables
2. Correlation matrices: Display correlations between multiple variables
3. Heatmaps: Colorful visualization of correlation matrices
4. Residual plots: Assess the fit of regression models and R²

## 10. Advanced Topics

1. Partial correlation: Correlation between two variables while controlling for other variables
2. Semi-partial correlation: Unique contribution of a variable to the total variance
3. Canonical correlation: Correlations between linear combinations of sets of variables
4. Intraclass correlation: Assesses rating reliability
5. Point-biserial correlation: Correlation between a continuous variable and a binary variable

## 11. Implementing in Python

Using NumPy and SciPy for correlation and R-squared calculations:

```python
import numpy as np
from scipy import stats

# Generate sample data
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 5])

# Calculate Pearson correlation
r, p_value = stats.pearsonr(x, y)
print(f"Pearson correlation coefficient: {r}")
print(f"P-value: {p_value}")

# Calculate R-squared
r_squared = r**2
print(f"R-squared: {r_squared}")

# Calculate Spearman correlation
rho, p_value = stats.spearmanr(x, y)
print(f"Spearman correlation coefficient: {rho}")

# Calculate Kendall's Tau
tau, p_value = stats.kendalltau(x, y)
print(f"Kendall's Tau: {tau}")
```

## 12. Conclusion

Correlation, R, and R² are powerful tools for understanding relationships between variables and assessing model fit. They provide valuable insights across various fields and are fundamental to many statistical analyses. However, their proper use requires a deep understanding of their assumptions, limitations, and interpretations.

As data analysis continues to evolve, these concepts remain crucial. They form the foundation for more advanced statistical techniques and machine learning models. By mastering these fundamental concepts, analysts and researchers can build more robust models, make more informed decisions, and gain deeper insights from their data.

It's important to remember that while these measures provide valuable information, they should always be used in conjunction with domain knowledge, careful data examination, and consideration of the broader context of the analysis. The true power of these tools lies not just in their calculation, but in their thoughtful application and interpretation.










lst sqr vs rotations / superimpose on original data.
fit / finding R^2
p-value for R^2

correlation / R / ....

how much better is my fit in terms of quantification / 
how good is my machine learning model / 
difference quantification for effectiveness analysis /
variation around mean /

( SSR black - SSR blue  ) / SSR black. --> div that makes R^2 a percentage. 
if val = x%, it means SSR blue has x% less variation.

variation against single attribute shows (data - mean) variation concept.
statistically, some variance = SSR / N = average SSR.

R^2 = introduction of Var(Attr1) - Var(LineFit) / Var(Attr1) = num.

R2 explains 'if i introduce a new attribute, how much % does Attr2 help me in reduction of variance around Attr1.'

since all variations are div by N, original formula transforms into: [ SS(Mean) - SS(Fit) ] / SS(Mean) 


adjusted R2 / / scale by parameters / /
introduction of parameters randomly brings chaos events and randomized probability of bias ( ex: what if my mice somehow have a 70% random head bias instead of tail, despite the coin flip having no sense? )

adj R2 scales with params. 


F = variation explained by attr1 / variation not explained by attr1. 