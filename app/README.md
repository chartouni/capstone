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

- `main.py`: Main Streamlit application entry point
- `pages/`: Multi-page app structure
  - `01_predict.py`: Single prediction page
  - `02_batch_predict.py`: Batch prediction page
  - `03_model_info.py`: Model performance and details
  - `04_feature_importance.py`: Feature importance dashboard
- `components/`: Reusable UI components
- `styles/`: Custom CSS styling

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
1. **Streamlit Cloud**: Deploy directly from GitHub
2. **AWS/Azure/GCP**: Deploy on cloud platforms
3. **Docker**: Containerized deployment
4. **On-premise**: Deploy on AUB servers

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
