# Network Intrusion Detection Using Machine Learning

## Overview
This project demonstrates the use of machine learning and neural networks to detect network intrusions by analyzing network traffic data. The aim is to identify various types of network attacks, including brute force attacks, SQL injection, and cross-site scripting (XSS), as well as other intrusion types, with high accuracy and generalization.

## Motivation
With the increasing reliance on networked systems, cybersecurity has become a critical focus area. This project explores the application of machine learning in intrusion detection systems (IDS) to automate the identification of malicious activities. Through experimentation with multiple models and approaches, this project aims to advance understanding and highlight the capabilities of machine learning in cybersecurity.

## Data Source
The data used in this project is the CICIDS2017 dataset, a benchmark dataset for intrusion detection research. It includes network traffic data with both normal behavior and a variety of attack scenarios. The dataset, along with its raw pcap files and preprocessed CSV files, is publicly available at the [University of New Brunswick's website](https://www.unb.ca/cic/datasets/ids-2017.html).

## Challenges
Initially, I explored multiple data sources, including pcap files and log files, but faced challenges with insufficient data or difficulty in loading due to size constraints. Specifically, I attempted to use pcap files from the CICIDS2017 dataset, which contain raw network traffic data from which the CSV files were derived. However, these files were extraordinarily large (e.g., over 50GB), and despite optimization attempts, loading even a single pcap file took over an hour and still did not complete due to the sheer volume of data. Additionally, I explored pcap files from other datasets, but many of them had insufficient data to train and evaluate a robust machine learning model effectively. 

As a result, I shifted to using the pre-extracted CSV files from the CICIDS2017 dataset, which provided a more efficient and manageable format for processing. To further streamline the process, I reduced the dataset size by selecting a portion of the data to balance runtime and performance without compromising the diversity of attack scenarios.

One limitation was the imbalanced representation of certain attack types like brute force, SQL injection, and XSS. These limitations point to potential improvements, such as simulating a virtual environment to generate balanced data for underrepresented attack types.

## Approach

### Models Tried

- Isolation Forest: Tested for anomaly detection but lacked granularity for classifying specific attack types.
- Random Forest: Established as a baseline, providing good accuracy but limited scalability.
- Custom Neural Network with TensorFlow: Developed and refined to handle complex patterns in the data, achieving better generalization.

### Data Preprocessing
- Feature Engineering: Created ratio-based, time-based, statistical, and interaction features to enhance model performance.
- Handling Imbalance: Addressed class imbalance with SMOTE and downsampling techniques.
![Imbalance Chart](https://github.com/kalebhings/Network-Activity-Detection/blob/main/balancingchart.png?raw=true)
- Standardization: Scaled features to ensure compatibility with neural network models.

### Model Optimization
- Learning Rate Scheduler: Used ReduceLROnPlateau to adjust learning rates dynamically.
- Early Stopping: Prevented overfitting by monitoring validation loss during training.
- Custom Metrics: Evaluated precision, recall, and F1-score for each attack type.

### Evaluation Metrics
The model's performance is assessed using:
- Precision
- Recall
- F1-Score
- Confusion Matrix Visualization

## Results
The model achieved an overall accuracy of 90% on the holdout set, performing well on classes with sufficient representation, such as DDoS (F1-Score: 0.98) and DoS Hulk (F1-Score: 0.97). These results indicate strong detection capabilities for these attack types.

However, performance dropped significantly for underrepresented classes like Brute Force, SQL Injection, and XSS, where low Precision (e.g., 0.01 for Brute Force) highlights issues with false positives. While Recall for these classes was often high (e.g., 1.00 for XSS), the imbalance in the dataset limited the model's ability to generalize effectively.

The disparity between well-represented and underrepresented classes demonstrates the need for additional data or advanced techniques to improve detection for rare attacks.

The code and it being ran can be viewed in this [notebook](https://github.com/kalebhings/Network-Activity-Detection/blob/main/NeuralNetworkModel.ipynb). 

![confusion matrix](https://github.com/kalebhings/Network-Activity-Detection/blob/main/confusionmatrix.png?raw=true)

## Insights
- Increasing data for underrepresented attack types is critical for improving recall and precision.
- Simulating a virtual environment for attack data collection could provide balanced, high-quality datasets to have more data for some of the attacks that were porly represented.
- Fine-tuning feature engineering and experimenting with ensemble methods may further improve detection capabilities.

## Future Work
- Set up a virtual lab to generate additional attack data for balanced datasets.
- Deploy the model to monitor live network traffic in a production environment.
- Explore advanced architectures like transformers or ensemble techniques for better accuracy.
- Integration with Raw PCAP Data: Developing a pipeline to process and extract features directly from PCAP files would allow the model to predict network intrusions in real-world scenarios. This approach eliminates the reliance on pre-cleaned and organized datasets, making the system more practical for live deployment.

## Tools and Technologies
- Programming Languages: Python
- Libraries: TensorFlow, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, SMOTE (Imbalanced-Learn)
- Dataset: CICIDS2017

## Acknowledgments
Thanks to the University of New Brunswick for providing the CICIDS2017 dataset.