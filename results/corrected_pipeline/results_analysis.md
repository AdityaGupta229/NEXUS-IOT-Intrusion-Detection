Results Analysis and Interpretation

1. Experimental Setup

The proposed Federated Learning-Based Privacy-Preserving IoT Intrusion Detection System was evaluated using the CICIoT2023 dataset.

Three models were compared:

1. Centralized PyTorch Model
2. IID Federated Averaging (FedAvg)
3. Non-IID Federated Averaging (FedAvg)

The federated experiments used four simulated clients. The IID experiment used approximately equal class distributions, whereas the Non-IID experiment used a Dirichlet distribution with alpha = 0.5.

All three models used the same MLP architecture, optimizer, learning rate, and five training rounds/epochs as applicable.

2. Final Test Performance

Model| Accuracy| Macro F1| Weighted F1
Centralized PyTorch| 77.83%| 51.60%| 77.11%
IID FedAvg| 74.94%| 45.68%| 72.74%
Non-IID FedAvg| 61.44%| 32.67%| 53.15%

3. Key Observations

- The Centralized PyTorch model achieved the highest overall accuracy and F1 scores.
- IID FedAvg performed better than Non-IID FedAvg across all three reported metrics.
- Non-IID data distribution made federated training more challenging in this experiment.
- Several frequent attack classes achieved strong classification performance.
- Some minority attack classes had low F1 scores, indicating difficulty in detecting underrepresented attack categories.

4. Interpretation

The results indicate that data distribution across clients has an important effect on federated model performance.

The IID federated model achieved performance closer to the centralized model than the Non-IID model. This suggests that heterogeneous client data can make global model optimization more challenging.

The difference between centralized and federated performance also indicates that additional training rounds, improved aggregation strategies, and client-level optimization may be worth investigating.

5. Limitations

- Federated clients were simulated on a single computer rather than deployed on separate physical IoT devices.
- Secure aggregation and differential privacy were not implemented.
- Therefore, the current prototype demonstrates federated training but does not provide a formal privacy guarantee.
- The training and validation datasets were sampled, while final testing used the original test dataset.
- Minority-class performance remains a limitation.
- The experiments used a CPU-based environment with limited computational resources.

6. Conclusion

The experiments demonstrate that Federated Averaging can be used to train an IoT intrusion detection model across simulated clients without directly combining their local training datasets.

IID FedAvg achieved better results than Non-IID FedAvg in the conducted experiments. However, the centralized model achieved the strongest overall test performance.

These findings highlight the importance of client data distribution and class imbalance in federated intrusion detection systems.