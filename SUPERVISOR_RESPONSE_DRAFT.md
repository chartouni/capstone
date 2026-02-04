# Response to Supervisor Feedback

## To: [Supervisor Name]
## Re: Progress Report Feedback - Citation Prediction Capstone

Dear [Supervisor Name],

Thank you for your feedback on my progress report. I'm addressing each of your suggestions below:

---

## 1. List All Features Used ✅ **Completed**

I've created a comprehensive feature documentation (`COMPLETE_FEATURE_LIST.md`) that includes:

- **All 5,019 features enumerated:**
  - 9 venue features (SJR, CiteScore, SNIP with percentiles and composites)
  - 10 author collaboration features (team size, institutional diversity, international collaboration)
  - 5,000 TF-IDF text features from abstracts

- **Feature importance rankings** showing venue prestige (SJR: 20.7%) and abstract content (50.5% aggregate) as top predictors

- **Ex ante validation documentation** confirming all features are observable at publication time

**Document attached:** `COMPLETE_FEATURE_LIST.md`

---

## 2. Include Samples from Other Institutions ⚠️ **Need Guidance**

This suggestion would significantly improve model generalization and potentially increase F1/accuracy. However, I'd like to discuss the scope and timeline implications:

### **Benefits:**
- Better generalization beyond AUB-specific patterns
- More robust model for broader applicability
- Potentially improved performance metrics
- Stronger contribution to the field

### **Implementation Challenges:**
- **Data Collection:** Need Scopus/SciVal exports for 10,000+ papers from other institutions (estimated 1-2 weeks for data acquisition and permissions)
- **Data Integration:** Merge with existing 14,832 AUB papers, handle potential format differences (3-5 days)
- **Retraining:** Re-run entire pipeline (feature engineering, model training, validation) (1 week)
- **Analysis:** Compare AUB-only vs multi-institution models, analyze institution-specific patterns (3-5 days)

**Total estimated time: 3-4 weeks**

### **Current Status:**
- Project is 85-90% complete with AUB-only data
- Models achieve 79% ROC-AUC (industry standard for citation prediction)
- All technical infrastructure is in place

### **Options:**

**Option A: Expand Dataset (Recommended if time permits)**
- Add papers from comparable regional universities (LAU, USJ, AUS)
- Retrain models and compare performance
- Timeline: 3-4 additional weeks

**Option B: Current Scope + Future Work Section**
- Complete project with AUB-only data (maintain current timeline)
- Document multi-institution expansion as "Future Work" in final report
- Demonstrate proof-of-concept with strong AUB-specific results
- Timeline: Maintain current 1-2 week completion estimate

**Option C: Limited Pilot**
- Add 2,000-3,000 papers from one other institution as validation
- Quick test of generalization without full retraining
- Timeline: 1-2 additional weeks

**My recommendation:** I'm inclined toward **Option B** given timeline constraints, but I'm flexible and would value your guidance on the best approach given the program's deadlines.

---

## 3. Share Cleansed Data & Code ✅ **Already Prepared**

All deliverables are ready to share:

### **Cleansed Data:**
- `cleaned_data.pkl` (14,832 papers, 68 features)
- `X_all.pkl` (5,019 engineered features)
- `X_train_temporal.pkl` / `X_test_temporal.pkl` (temporal split)
- `y_classification.pkl` / `y_regression.pkl` (target variables)

### **Code Repository:**
- Complete GitHub repository with documented codebase
- 14 Jupyter notebooks (data exploration, feature engineering, model training, analysis)
- Reusable Python modules (`src/data/`, `src/features/`, `src/models/`)
- Streamlit deployment application

### **Documentation:**
- `README.md` (project overview and setup)
- `PROJECT_STRUCTURE.md` (architecture documentation)
- `COMPLETE_FEATURE_LIST.md` (feature enumeration and validation)
- Notebook-level documentation with markdown cells

**Data sharing format:** Can provide as:
- GitHub repository link (code + documentation)
- Compressed data files (CSV + pickle formats)
- Any specific format you prefer

**Note on data sharing:** The Scopus/SciVal data is institution-licensed. I'll coordinate with AUB library to ensure proper data sharing permissions.

---

## Questions for You:

1. **Multi-institution data:** Which option (A/B/C) aligns best with program timeline and your expectations?

2. **Data format:** Do you have a preferred format for the cleansed data delivery? (CSV, Excel, pickle, database export?)

3. **Code repository:** Should I prepare a specific branch/release for final submission, or is the current state sufficient?

4. **Feature list:** Is the `COMPLETE_FEATURE_LIST.md` document sufficient, or would you like additional feature documentation (e.g., correlation matrices, distribution plots)?

---

## Next Steps (Pending Your Guidance):

**If Option B (Current Scope):**
- Finalize visualization generation (8 figures)
- Complete final capstone report (15-20 pages)
- Prepare defense presentation (12-15 slides)
- Package data and code for delivery
- **Timeline:** 1-2 weeks to completion

**If Option A (Multi-Institution):**
- Coordinate data collection from other institutions
- Implement expanded dataset pipeline
- Comparative analysis (AUB-only vs multi-institution)
- Updated final report with generalization analysis
- **Timeline:** 3-4 weeks to completion

I'm happy to discuss these options at your convenience. Please let me know your preferred direction, and I'll adjust the project plan accordingly.

Thank you for your continued guidance.

Best regards,
Mario Chartouni
