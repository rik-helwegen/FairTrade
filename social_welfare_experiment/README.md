Social welfare experiment
===============

This repository contains an implementation of the FairTrade method, applied for detecting unlawful social welfare.
The scripts assume access to a CSV version of the data. Due to privacy constraints the data for this experiment
cannot be made publicly available.

### utils.py
Helper functions. In specific; the data loader, the CEVAE network classes and the visualisation scripts.

### baselines.py
Obtains baseline accuracy for a Linear Regression, Random Forest and a Multi Layer Perceptron.    

### train_cevae.py
Trains a CEVAE model, in order to learn the causal mechanisms in the data, according to the assumed graph.

### train_aux.py
Train auxiliary models, for which the input is depends on the fairness constraints. A number of different
models are trained in this script in order to compare their relation in accuracy and sensitive inequality

### black_box_eval.py
Experiment set-up of black box evaluation method for causality-based fairness metrics.
Experiment stops at the sanity check of comparing the predictive performance on the original data and
counterfactual data.
