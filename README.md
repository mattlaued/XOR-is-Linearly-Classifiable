# Revisiting Non-separable Classification and its Applications in Anomaly Detection

The inability to linearly classify XOR has motivated much of deep learning.
We revisit this age-old problem and show that *linear* classification of XOR is indeed possible.
Instead of separating data between halfspaces, we propose a slightly different paradigm, *equality separation*, that adapts the SVM objective to distinguish data within or outside the margin.
Our classifier can then be integrated into neural network pipelines with a smooth approximation.
From its properties, we intuit that equality separation is suitable for anomaly detection.
To formalize this notion, we introduce *closing numbers*, a quantitative measure on the capacity for classifiers to form closed decision regions for anomaly detection.
Springboarding from this theoretical connection between binary classification and anomaly detection, we test our hypothesis on supervised anomaly detection experiments, showing that equality separation can detect both seen and unseen anomalies.

## Code Directory

Part of this submission includes code used to run the experiments in the main paper and appendix. The main contributions focus on applying the bump activation function as a smooth approximation for hyperplane learning in anomaly detection.


For linear binary classification experiments, we share our code for solving the XOR problem, general
linearly non-separable problems and linearly separable problems. 

For non-linear synthetic anomaly detection (AD) experiments, we share our code that uses shallow models, deep models with 2 hidden layers and deep models with 3 hidden layers. 

For experiments on non-linear supervised AD with NSL-KDD dataset, we share our code on shallow methods/baselines, binary classification, Negative Sampling (NS) and Deep Semi-Supervised Anomaly Detection (SAD).
Note that you will first need to download the NSL-KDD dataset.
We also include comparisons to halfspace separation and RBF separation for Thyroid and MVTec dataset.