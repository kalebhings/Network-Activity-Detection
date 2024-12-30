# Network Intrusion Detection Using Machine Learning

## Overview
This project demonstrates the use of machine learning and neural networks to detect network intrusions by analyzing network traffic data. The goal is to identify various types of network attacks, including brute force attacks, SQL injection, and cross-site scripting (XSS), among others, with high accuracy and generalization.

## Motivation
With the increasing reliance on networked systems, ensuring security is more critical than ever. This project explores the application of machine learning in cybersecurity to detect malicious activity in network traffic effectively. By leveraging advanced machine learning techniques and neural networks, the project aims to contribute to the growing field of automated intrusion detection systems.

## Data Source
The data used in this project comes from the CICIDS2017 dataset, a comprehensive dataset designed for evaluating intrusion detection systems. The dataset includes network traffic from a variety of attack scenarios and normal behavior. It is publicly available at the [University of New Brunswick's website](https://www.unb.ca/cic/datasets/ids-2017.html).

## Approach
1. **Data Preprocessing**: 
   - Raw data from CSV files is cleaned, scaled, and encoded for use in machine learning models.
   - Missing and infinite values are handled to ensure robustness.

2. **Model Development**:
   - A Random Forest classifier was initially implemented to establish a baseline.
   - A neural network using TensorFlow was developed to improve performance and handle complex patterns in the data.

3. **Optimization**:
   - Early stopping and learning rate scheduling are used to prevent overfitting and improve model generalization.
   - Class imbalances are addressed using techniques like SMOTE and class weighting.

## Key Features
- **Neural Network**: A deep learning model with multiple dense layers and dropout for regularization.
- **Early Stopping and ReduceLROnPlateau**: Ensures efficient training by stopping early and dynamically adjusting the learning rate.
- **Feature Engineering**: Comprehensive preprocessing pipeline to handle raw network traffic data.
- **Confusion Matrix and Metrics**: Visual and quantitative evaluation of the model's performance across multiple attack types.

## Results
The project achieves high accuracy in detecting various types of network intrusions. Metrics such as precision, recall, and F1-score are evaluated for each attack type to ensure comprehensive assessment.

## Tools and Technologies
- Python
- TensorFlow
- Pandas and NumPy
- Scikit-learn
- Seaborn and Matplotlib
- SMOTE (Synthetic Minority Oversampling Technique)

## Data Disclaimer

The CICIDS2017 dataset is publicly available and was used in compliance with its licensing terms. Ensure proper attribution if reusing the dataset for derivative work.

## Future Work
- Integration with real-time network monitoring tools.
- Exploration of other machine learning algorithms like gradient boosting and unsupervised anomaly detection.
- Deployment of the trained model for live traffic analysis in a production environment.

## Acknowledgments
Special thanks to the University of New Brunswick for providing the CICIDS2017 dataset and supporting the research community in intrusion detection.