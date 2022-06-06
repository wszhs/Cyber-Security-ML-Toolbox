# Cyber Security Machine Learning
![Python version](https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8-blue.svg)
![Image file ](images/logo.jpg)

**Trusted-AI for Cyber Security**

## Adversarial Machine Learning for Cyber Security

The robustness test package of csmt provides a series tools that enable researchers to defend and evaluate Machine Learning models and applications against adversarial threats. 
In particular, csmt supports popular machine learning frameworks (TensorFlow, Keras, PyTorch, scikit-learn, XGBoost, LightGBM, CatBoost, etc. ), as well as multiple adversarial attack and defense strategies.
More importantly, we employ restricted feature-space attacks using remapping functions and implement end-to-end problem-space attacks in Network Intrusion Detections(NIDSs) and Windows malware detections.

### Base Modules

<details>
  <summary><b>Data Modules</b></summary>
  csmt.datasets
  <ul>
    <li><b>Kitsune </b>
    <li><b>NSLKDD </b>
    <li><b>CICIDS2017 </b>
    <li><b>CICIDS2018 </b>
    <li><b>CICAndMal2017 </b>
    <li><b>CTU13 </b>
    <li><b>DOHBRW </b>
    <li><b>NSLKDD </b>
    <li><b>MalImg </b>
    <li><b>TwitterSpam</b>
    <li><b> Androzoo </b>
    <li><b>DreBin </b>
    <li><b>Contagiopdf </b>
</ul>
    csmt.datasets.graph
      <ul>
    <li><b>Bitcoin-Alpha </b>
    <li><b>Tencent-Weibo</b>
    <li><b>Elliptic</b>
    <li><b>Yelp-Chi</b>
    </ul>
</details>

<details>
  <summary><b>Data Processing Modules</b></summary>
  csmt.data_reduction
  <ul>
    <li><b>Alipy </b>
    <li><b>ModAL </b>
</ul>
  csmt.data_validation
      <ul>
    <li><b> Deepchecks </b>
    <li><b>Evidently</b>
    </ul>
</details>

<details>
  <summary><b>Model Modules</b></summary>
  csmt.classifiers.classic
  <ul>
    <li><b>CatBoost </b>
    <li><b>MaMPF </b>
    <li><b>DecisionTree </b>
    <li><b>DeepForest </b>
    <li><b>HMM </b>
    <li><b>KNearestNeighbours </b>
    <li><b>LightGBM </b>
    <li><b>LogisticRegression </b>
    <li><b>NaiveBayes </b>
    <li><b>RandomForest </b>
    <li><b>SupportVectorMachine</b>
    <li><b> XGBoost </b>
</ul>
    csmt.classifiers.anomaly_detection
    <ul>
    <li><b>KitNet </b>
    <li><b>DIFF-RF </b>
    <li><b>AutoEncoder </b>
    <li><b>IsolationForest </b>
    <li><b>OCSVM </b>
    <li><b>HBOS </b>
    <li><b>VAE</b>
    </ul>
csmt.classifiers.keras
    <ul>
    <li><b>MLP </b>
    <li><b>LSTM </b>
    <li><b>RNN </b>
    </ul>
    csmt.classifiers.torch
    <ul>
    <li><b>Transformer </b>
    <li><b>CNN(MalCov) </b>
    <li><b>LSTM </b>
    <li><b>RNN </b>
    <li><b>MLP </b>
    <li><b>FS-Net</b>
    </ul>
csmt.classifiers.ensemble
<ul>
    <li><b>HardEnsemble </b>
    <li><b>SoftEnsembleModel </b>
    <li><b>StackingEnsembleModel </b>
    <li><b>BayesEnsembleModel </b>
    </ul>
    csmt.classifiers.graph.model
<ul>
    <li><b>GCN </b>
    <li><b>GraphSAGE </b>
    <li><b>GIN </b>
    <li><b>GAT</b>
    </ul>
</details>

<details>
  <summary><b>Attack and Defense Modules</b></summary>
  csmt.attacks
  <ul>
    <li><b>Base Class Attacks </b>
    <li><b>Base Class Evasion Attacks </b>
    <li><b>Base Class Poisoning Attacks </b>
    <li><b>Base Class Inference Attacks </b>
</ul>
    csmt.attacks.evasion
    <ul>
    White-Box Attack
    <li><b> Carlini and Wagner L_0 Attack </b>
    <li><b>Carlini and Wagner L_2 Attack</b>
    <li><b>Carlini and Wagner L_inf Attack</b>
    <li><b>DeepFool</b>
    <li><b>Fast Gradient Method (FGM)</b>
    <li><b>Basic Iterative Method (BIM)</b>
    <li><b>Projected Gradient Descent (PGD)</b>
    <li><b>Jacobian Saliency Map Attack (JSMA)</b>
    </ul>
    <ul>
    Black-box Attack
    <li><b>Zeroth-Order Optimization (ZOO) Attack</b>
    <li><b>Natural Evolutionary Strategies</b>
    <li><b>ZO Stochastic Gradient Descent</b>
    <li><b>HopSkipJump Attack</b>
    <li><b>Boundary Attack</b>
    <li><b>Genetic Attack</b>
    <li><b>Differential Evolution Attack</b>
    <li><b>Particle Swarm Attack</b>
    <li><b>BayesOpt Attack</b>
    </ul>
    csmt.attacks.graph
    <ul>
    Graph Modification Attack
    <li><b> DICE </b>
    <li><b>STACK</b>
    <li><b>NEA</b>
    <li><b>Nettack</b>
    <li><b>TopologyAttack</b>
    <li><b>RL-S2V</b>
    <li><b>FGA</b>
    </ul>
    <ul>
    Black-box Attack
    <li><b>TDGIA</b>
    <li><b>SPEIT</b>
    <li><b>GA2C</b>
    </ul>
  csmt.estimators
  <ul>
    <li><b> Base Class Trainer </b>
    <li><b>Base Class KerasEstimator </b>
    <li><b>Base Class PyTorchEstimator </b>
    <li><b>Base Class ScikitlearnEstimator </b>
    <li><b>Base Class TensorFlowEstimator </b>
    <li><b>Base Class TensorFlowV2Estimator </b>
</ul>
csmt.defences.trainer
  <ul>
    <li><b> Base Class Trainer </b>
    <li><b> Adversarial Training</b>
    <li><b>Ensemble Adversarial Training </b>
    <li><b>Bayesain Ensemble Adversarial Training</b>
    <li><b>Transfer Ensemble Adversarial Training </b>
    <li><b>Nash Ensemble Adversarial Training </b>
</ul>
csmt.estimators
  <ul>
    <li><b> Base Class Estimator </b>
    <li><b> Base Class KerasEstimator</b>
    <li><b>Base Class PyTorchEstimator </b>
    <li><b>Base Class ScikitlearnEstimator</b>
    <li><b>Base Class TensorFlowEstimator </b>
    <li><b>Base Class TensorFlowV2Estimator </b>
</ul>
</details>

<details>
  <summary><b>Domain Constraints Modules</b></summary>
   Restricted Feature Space
  <ul>
    <li><b>Remapping Function (mask) </b>
    <li><b>Learning constraints </b>
</ul>
</details>

The details of **Domain Constraints** can be found [here](https://github.com/wszhs/my_knowledge_map/blob/master/adversarial_ml/cyberSecurity/cybersecurity_aml.md)

### End-to-end Problem-Space Modules

<details>
  <summary><b>Traffic</b></summary>
  Feature Extractor
  <ul>
    <li><b>AfterImage </b>
    <li><b>CICFlowmeter </b>
</ul>
  csmt.ps_attack.packet_attack
  <ul>
    <li><b>Random Attack </b>
    <li><b>GA Attack </b>
    <li><b>PSO Attack </b>
    <li><b>RL Attack </b>
    <li><b>ZOSGD Attack </b>
    <li><b>BayesAttack </b>
    (Black box attack algorithms in the basic attack module can be customized the problem-space attack)
</ul>
  csmt.ps_attack.traffic （traffic manipulation）
  <ul>
    <li><b>Change the duration </b>
    <li><b>Change the time interval</b>
    <li><b>Change the packet length </b>
    <li><b>Add the new packets </b>
</ul>
</details>

<details>
  <summary><b>Window PE</b></summary>
  csmt.attacks.evasion.pe_malware_attack
  <ul>
    <li><b>Append based attacks</b>
    <li><b>Section insertion attacks </b>
    <li><b>Slack manipulation attacks </b>'
    <li><b>DOS Header Attacks</b>
</ul>
</details>

The details of **NIDSs Attack** can be found [here](https://github.com/wszhs/my_knowledge_map/blob/master/adversarial_ml/cyberSecurity/NIDS/NIDS_aml.md)

The details of **Windows PE Attack** can be found [here](https://github.com/wszhs/my_knowledge_map/blob/master/adversarial_ml/cyberSecurity/malware/PE_aml.md)

***
## Explainable Machine Learning for Cyber Security
The explainability module of csmt supports interpretability and explainability of datasets and machine learning models for Cyber Security. It includes a comprehensive set of algorithms that cover different dimensions of explanations along with proxy explainability metrics.

### Supported Explainability Algorithms
<details>
  <summary><b>Explain Modules</b></summary>
  csmt.interpretability
  <ul>
    <li><b>LIME </b>
    <li><b>DeepLIFT </b>
    <li><b>Permutation Test </b>
    <li><b>KernelSHAP </b>
    <li><b>Shapley Effects </b>
    <li><b>TreeSHAP </b>
    <li><b>LossSHAP </b>
    <li><b>SAGE </b>
</ul>
</details>

### Evaluation Metrics to Explainability algorithms
<details>
  <summary><b>Faithfulness</b></summary>
quantifies to what extent explanations follow the predictive behaviour of the model (asserting that more important features play a larger role in model outcomes)
 <br><br>
  <ul>
    <li><b>Faithfulness Correlation </b>
    <li><b>Faithfulness Estimate </b>
    <li><b>Monotonicity Metric </b>
</ul>
</details>

<details>
  <summary><b>Robustness</b></summary>
measures to what extent explanations are stable when subject to slight perturbations of the input, assuming that model output approximately stayed the same
 <br><br>
  <ul>
    <li><b>Max-Sensitivity </b>
    <li><b>Avg-Sensitivity </b>
</ul>
</details>

<details>
  <summary><b>Complexity</b></summary>
captures to what extent explanations are concise i.e., that few features are used to explain a model prediction
 <br><br>
  <ul>
    <li><b>Sparseness </b>
    <li><b>Complexity </b>
    <li><b>Effective Complexity </b>
</ul>
</details>

***
## User Cases

### ETA: Explainable and Transferable Adversarial Attack for ML-Based Network Intrusion Detectors
[ETA](ETA.md) is a general Explainable Transfer-based black-box adversarial Attack framework aiming to 1) craft transferable adversarial examples across various types of ML models and 2) explain why adversarial examples and adversarial transferability exist in NIDSs.
  
### CARE: Ensemble Adversarial Robustness Evaluation Against Adaptive Attackers for Security Applications
[CARE](CARE.md) is an ensemble adversarial robustness evaluation platform for the cybersecurity domain.

### BayesAttack: Query-Efficient End-to-end Adversarial Attacks against Security Detectors via Bayesian Optimisation
[BayesAttack](BayesAttack.md) is a general adversarial attack framework for security applications aiming to address two critical requirements in practice: 1) achieving query-effective black box attacks, 2) directly modifying the original entities, and meeting the constraints of the problem space.

***
## Acknowledgements
csmt is built with the help of several open source packages. All of these are listed in setup.py and some of these include:
- [Tensorflow]()
- [scikit-learn]()
- [AI Explainability 360]()
- [adversarial-robustness-toolbox]()
- [Quantus]()
- [SHAP]()
- [Evidently]()
- [deepchecks]()
- [ALiPy]()


