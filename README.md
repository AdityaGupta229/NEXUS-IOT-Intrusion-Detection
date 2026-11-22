NEXUS

Federated Learning-Based Privacy-Preserving IoT Intrusion Detection System

NEXUS is a machine learning-based IoT security project designed to identify suspicious network traffic and detect potential cyberattacks using centralized and federated learning approaches.

The project includes a web-based dashboard that allows users to upload network traffic data, select a trained model, analyze traffic, and visualize prediction results.

---

Project Overview

The Internet of Things (IoT) connects devices such as smart sensors, cameras, gateways, and industrial equipment. As the number of connected devices increases, network security becomes increasingly important.

NEXUS explores Federated Learning (FL) as an alternative to conventional centralized model training. In the simulated federated setup, multiple clients train models locally and share model updates for aggregation.

The project compares three approaches:

- Centralized Learning
- IID Federated Learning using FedAvg
- Non-IID Federated Learning using FedAvg

The trained models are integrated into a Flask dashboard for traffic analysis and intrusion detection.

Objectives

- Detect normal and malicious IoT network traffic.
- Implement and compare centralized and federated learning.
- Study the effect of IID and Non-IID client data distributions.
- Provide an interactive dashboard for CSV-based traffic analysis.
- Display prediction summaries, charts, and model performance metrics.

Technology Stack

Component| Technology
Programming Language| Python
Machine Learning| PyTorch, Scikit-learn
Data Processing| Pandas, NumPy
Backend| Flask
Frontend| HTML, CSS, JavaScript
Visualization| Chart.js
Model Serialization| Joblib, PyTorch
Dataset| CICIoT2023

Dataset

The project uses the CICIoT2023 dataset for IoT intrusion detection.

The dataset contains network traffic features and multiple traffic classes, including benign traffic and different attack categories.

Dataset source:

https://www.unb.ca/cic/datasets/iotdataset-2023.html

The project uses processed training and validation samples, federated client datasets, and a separate test dataset for model evaluation.

System Workflow

1. Load and preprocess the network traffic dataset.
2. Train a centralized machine learning model.
3. Divide the training data among simulated federated clients.
4. Train IID and Non-IID federated models using FedAvg.
5. Evaluate the trained models on the test dataset.
6. Load the saved models into the Flask application.
7. Upload a CSV file through the dashboard.
8. Select a model and run traffic analysis.
9. View predicted classes, traffic status, confidence estimates, charts, and results.

Model Performance

The following results were obtained during the project's final test-set evaluation.

Model| Accuracy| Macro F1| Weighted F1
Centralized| 77.83%| 0.5160| 0.7711
IID FedAvg| 74.94%| 0.4568| 0.7274
Non-IID FedAvg| 61.44%| 0.3267| 0.5315

These results demonstrate the performance of the implemented models on the selected test dataset. Performance may vary with dataset distribution and traffic characteristics.

Dashboard Features

- Overview dashboard
- CSV-based traffic analysis
- Centralized, IID FedAvg, and Non-IID FedAvg model selection
- Normal and attack traffic summary
- Traffic distribution visualization
- Attack class distribution visualization
- Model performance comparison
- Detection results table
- Project information and workflow page

Project Structure

Major Project/
├── dashboard/
│   ├── templates/
│   └── static/
│       ├── css/
│       ├── js/
│       └── images/
├── dataset/
│   ├── raw/
│   └── processed/
├── federated/
├── models/
│   └── saved/
├── preprocessing/
├── notebooks/
├── app.py
├── requirements.txt
└── README.md

Installation and Setup

1. Clone the repository

git clone <repository-url>
cd Major-Project

2. Create a virtual environment

python -m venv venv

3. Activate the environment

Windows PowerShell:

.\venv\Scripts\Activate.ps1

4. Install dependencies

pip install -r requirements.txt

5. Run the application

python app.py

6. Open the dashboard

Visit:

http://127.0.0.1:5000

Make sure the trained model files and preprocessing artifacts are available in the expected project directories before running the application.

Limitations

- Federated clients are simulated on a single machine.
- The current implementation does not use secure aggregation or differential privacy.
- Federated Learning is explored as a privacy-oriented training approach; formal privacy guarantees are not claimed.
- Predictions depend on the quality and distribution of the input data.
- The Flask development server is intended for local demonstration and testing.

Future Enhancements

- Deploy the dashboard to a production-ready environment.
- Support larger datasets through optimized batch processing.
- Add additional evaluation metrics and explainability features.
- Explore secure aggregation and differential privacy.
- Extend the system to real-time IoT traffic monitoring.

Conclusion

NEXUS demonstrates the implementation and comparison of centralized and federated learning approaches for IoT intrusion detection. The project combines machine learning, simulated federated training, and an interactive web dashboard to provide a practical environment for analyzing network traffic and studying model performance.

---

Project: NEXUS
Domain: Artificial Intelligence, Machine Learning, IoT Cybersecurity
Purpose: Academic Major Project