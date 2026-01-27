# Streamlit Deployment App

This directory contains the Streamlit web application for the citation prediction tool.

## Purpose

Deploy trained models as an interactive web application that allows AUB to:
- Upload new publication data
- Get citation predictions in real-time
- Visualize prediction confidence and feature contributions
- Compare multiple publications
- Track predictions over time

## Files

- `main.py`: Main Streamlit application entry point (landing page)
- `pages/`: Multi-page app structure
  - `01_🎯_Single_Prediction.py`: Single prediction interface
  - `02_📁_Batch_Prediction.py`: Batch prediction with CSV upload
  - `03_📈_Model_Performance.py`: Model metrics and evaluation

## Features

### 1. Single Publication Prediction
- Input publication metadata manually or via form
- Get instant citation prediction (classification + regression)
- View confidence scores and prediction intervals
- See feature contributions to prediction

### 2. Batch Prediction
- Upload CSV file with multiple publications
- Process batch predictions
- Download results with predictions
- View summary statistics

### 3. Model Information
- Display model performance metrics
- Show training/validation curves
- Compare different models
- View temporal validation results

### 4. Feature Importance
- Interactive feature importance plots
- Filter by feature categories
- Compare importance across models
- SHAP value visualizations

## Prerequisites

Before running the app, you need to have trained models and feature artifacts:

1. **Trained Models** (from notebooks 30-31):
   - `models/classification/logistic_regression.pkl`
   - `models/classification/random_forest.pkl`
   - `models/classification/xgboost.pkl`
   - `models/classification/lightgbm.pkl`
   - `models/regression/random_forest.pkl`
   - `models/regression/xgboost.pkl`
   - `models/regression/lightgbm.pkl`

2. **Feature Artifacts** (from notebooks 20-23):
   - `data/features/tfidf_vectorizer.pkl` (Required)
   - `data/features/venue_statistics.pkl` (Optional but recommended)

3. **Performance Metrics** (from notebooks 30-31):
   - `reports/metrics/classification_results.csv` (Optional, for performance page)
   - `reports/metrics/regression_results.csv` (Optional, for performance page)

**To generate these files:**
```bash
# Run the feature engineering notebooks
jupyter notebook notebooks/20_feature_engineering_text.ipynb
jupyter notebook notebooks/21_feature_engineering_venue.ipynb
jupyter notebook notebooks/22_feature_engineering_author.ipynb
jupyter notebook notebooks/23_feature_engineering_final.ipynb

# Run the model training notebooks
jupyter notebook notebooks/30_classification_models.ipynb
jupyter notebook notebooks/31_regression_models.ipynb
```

## Running the App

### Locally:
```bash
# From project root
streamlit run app/main.py

# Or with specific configuration
streamlit run app/main.py --server.port 8501
```

### With Docker:
```bash
# Build image
docker build -t citation-predictor .

# Run container
docker run -p 8501:8501 citation-predictor
```

## Configuration

Configure app settings in `config/config.yaml`:
- Port and host
- Theme and styling
- Model paths
- Feature names and descriptions

## Input Format

The app expects the following input fields:
- **Title**: Publication title
- **Abstract**: Publication abstract text
- **Authors**: List of authors with h-index and citation counts
- **Venue**: Publication venue name
- **Year**: Publication year
- **Field**: Research field/domain

## Output Format

Predictions include:
- **Classification**: High-impact (top 25%) or not
- **Confidence**: Probability of high-impact class
- **Regression**: Predicted citation count with confidence interval
- **Feature Contributions**: Top features driving the prediction
- **Similar Papers**: Similar historical papers for context

## Deployment

### Production deployment options:

1. **Streamlit Cloud**:
   - Push your code to GitHub
   - Connect your repository to [Streamlit Cloud](https://streamlit.io/cloud)
   - Note: Model files need to be available (consider using Git LFS or external storage)

2. **Docker**:
   ```bash
   # Create Dockerfile (see project root)
   docker build -t citation-predictor .
   docker run -p 8501:8501 citation-predictor
   ```

3. **Cloud Platforms (AWS/Azure/GCP)**:
   - Deploy as a containerized app
   - Use cloud storage for models (S3, Azure Blob, GCS)

4. **On-premise (AUB Servers)**:
   - Copy project to server
   - Install dependencies: `pip install -r requirements.txt`
   - Run with `streamlit run app/main.py --server.port 8501`

## Maintenance

- **Model Updates**: Replace model files in `models/` directory
- **Feature Updates**: Update feature engineering pipeline
- **UI Updates**: Modify Streamlit pages and components
- **Monitoring**: Track prediction requests and performance

## Notes

- Models are loaded once at startup for efficiency
- Predictions are cached for repeated queries
- File uploads are limited to prevent abuse
- Session state maintains user interactions
