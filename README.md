Brief Summary of Capstone Project
Citation counts serve as a critical metric for assessing research impact, influencing funding decisions, academic promotions, and institutional rankings. However, predicting which papers will become highly cited remains challenging due to the complex interplay of author reputation, venue prestige, content quality, and field-specific dynamics. This project addresses the problem of early prediction of citation impact using only information available at publication time—before any citations accumulate.

For this capstone, I will be coordinating directly with AUB to access publication data and understand institutional research priorities. The university is interested in leveraging machine learning to identify emerging high-impact research, optimize resource allocation, and enhance its research visibility. This collaboration provides access to thousands of AUB publications with complete metadata, enabling domain-specific model development and validation.

Can we accurately predict citation counts and identify high-impact papers using machine learning models trained on publication metadata, author reputation metrics, and textual features?

Early identification of impactful research enables institutions to strategically allocate resources, helps researchers make informed venue selection decisions, and assists funding agencies in supporting promising work. For AUB specifically, this tool can highlight emerging high-impact research and guide institutional research strategy.

I’ll implement two complementary approaches: classification to identify papers likely to become highly cited, and regression to predict exact citation counts with strong ranking ability.

Company

Founded in 1866, AUB is a leading research university in the Middle East with over 9,000 students and 800+ faculty members. The university produces significant research output across diverse fields, including medicine, engineering, computer science, business, and social sciences. AUB is ranked among the top universities in the Arab region and maintains strong international research collaborations.

For this capstone, I am coordinating directly with AUB to access publication data and understand institutional research priorities.
Data
I expect to get metadata for over 10,000 papers with multiple features, including abstracts, citation counts, author names, h-index, etc.

Project Objective
Primary Objectives:

Develop High-Accuracy Citation Prediction Models:

Build classification models to identify highly-cited papers
Build regression models to predict citation counts with strong ranking correlation
Objective: Provide practical tools for early identification of high-impact research
Compare traditional ML (Logistic Regression, Random Forest) vs gradient boosting (XGBoost, LightGBM) or even deep learning

Test both classification and regression paradigms
Objective: Identify optimal algorithm-task combinations for citation prediction
Conduct Comprehensive Feature Importance Analysis

Rank features by predictive power using tree-based importance scores
Objective: Understand what drives citation impact in different research domains
Develop Interpretable Prediction Framework

Provide explanations for individual predictions (feature contributions)
Create visualizations for model performance and feature importance
Objective: Ensure models are trustworthy and actionable for decision-making
Project Methodology
Phase 1: Literature Review

Review citation prediction literature

Identify gap

Establish baseline metrics
●
Phase 2: Data Collection

To be coordinated with Khaled Noubani
●
Phase 3: Data Cleaning & Preprocessing

Quality Filters: Removed garbled text (encoding corruption), non-English papers (if any), invalid author names, missing critical fields

Tools: Python (pandas), custom cleaning scripts with regex pattern matching (if needed)

Storage: JSON format (for models) + CSV format (for inspection)

Phase 4: Feature Engineering

Author Features: Extract h-index (if possible) and citation counts from nested author objects, compute max/mean/sum statistics
Venue Features: Manual curation of 50+ top venue scores, computed historical citation statistics per venue
Text Features: TF-IDF vectorization
Handling Missing Data
Phase 5: Model Development and Comparison
● Algorithms: Logistic Regression, Random Forest, XGBoost, LightGBM
● Classification: Binary (top 25% vs rest), 5-fold stratified cross-validation
● Regression: Log-transformed citation counts, 5-fold cross-validation
● Tools: scikit-learn, pytorch
Compare performance metrics across datasets
● Analyze feature importance differences (which features matter in medical vs CS papers)
● Conduct error analysis
● Test temporal generalization: train on 2015-2017, test on 2018-2020
● Tools: pandas, matplotlib, seaborn for visualizations
Phase 6: Visualization & Reporting
Phase 6: Visualization & Reporting

ROC curves for classification performance
Confusion matrices and precision-recall curves
Feature importance bar charts (top 20 features)
Citation distribution histograms (actual vs predicted)
Tools: matplotlib, seaborn, plotly (optional for interactive visualizations)

I will be working on Jupyter notebooks. Two files were sent to me. They are too large to be uploaded here.
I must use the column containing the EID, which is same in both files, to match each article, and add the abstracts from the Scopus file to the corresponding entries in the SciVal file.
