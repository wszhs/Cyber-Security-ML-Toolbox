<!-- # Practical End-to-end Adversarial Attacks on Security Detectors using efficient query mechanisms -->
# Query-Efficient End-to-end Adversarial Attacks against Security Detectors via Bayesian Optimisation

## Background
Machine learning (ML) and Deep Learning (DL) are widely used in all types of security-related applications due to their superior performance and ability to detect unforeseen threats. 
However, machine learning models are susceptible to adversarial attack, which makes them difficult to deploy on a large scale.
Existing adversarial attack methods fail to satisfy practical requirements in security domains, as the actual adversarial attack is a query-effective black-box problem-space attack.

## Introduction

Researchers concluded that real-world adversarial attacks have to meet three conditions: 
- black-box attacks should be designed for real situations in which the attacker has no knowledge of the target model and can interact with it only by querying it. 
- due to the risk of being detected by defense systems or high inherent costs associated with model evaluation, query efficiency is highly prioritized when the damage caused by an attack is high.
- attackers should modify problem-space entities directly instead of feature-space entities.
Therefore, real-world adversarial attacks are therefore a query of an effective end-to-end black box attack of the problem-space.

In this paper, we propose BayesAttack, a general adversarial attack framework aiming to address two critical requirements in practice: 1) achieving a query-effective black-box attack, 2) directly modifying the original entities, and meeting the constraints of the problem-space.
By continuously and directly changing the entities of the problem space, we query the changes of the black-box model, and finally, by using Bayesian optimization, we find adversarial examples that satisfy the constraints faster.
Bayesian optimization can meet the needs of query-efficient black-box attacks, but problem space attacks are discrete, and native Bayesian optimization is better suited for continuous space optimization.
To solve this problem, we improve the naive Bayesian optimization method and design different kernel functions,  different acquisition functions, and optimization functions for different models.
we apply BayesAttack to three different types of security-related ML systems, and extensively evaluate our adversarial attack method against other black-box optimization methods.
Experimental results show that BayesAttack can provide high-quality real-world adversarial examples against ML models while meeting several special requirements in security domains.
We also explore how effective it is to assess the sensitivity of different models through practical adversarial attacks, which can help security operators to understand model decisions, diagnose system mistakes, give feedback to models, and reduce false positives.


## Evaluation

### Security Detectors

- Tabular Data based Systems.
Tabular data is the most common type that is structured into rows (also called feature vectors), each of which contains informative features about one sample. In general, most types of DNNs are designed to process tabular data. 

- Time-Series based Systems.
Time-series is a sequence of data samples indexed in time order. Any data with temporal information can essentially be represented as time-series, such as network traffic and system logs.

- Graph Data based Systems.
Graph data structure is useful for modeling a set of objects (nodes) and their relationships (links) at the same time. 


### Attack Baseline
- Random
- RL-s2v
- GA
- PSO
- NAS
- GradArgMax
- GAN




