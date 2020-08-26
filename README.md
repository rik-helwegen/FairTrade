# FairTrade method

A fairness method for machine learning models.
The goal of the FairTrade method is to include causality-based fairness constraints in the optimization of predictive models, also when unobserved confounders are present.


The full explanation of the method, including derivations, can be found in our research paper. Note that the method depends on context-specific assumptions and approximations, which are elaborated on in the paper.

<em>[Improving Fair Predictions Using Variational Inference In Causal Models](https://arxiv.org/abs/2008.10880)</em>

Authors Rik Helwegen, Christos Louizos and Patrick Forré.


## Content

The repository includes three experiments. For gaining understanding of the method, the simulation and IHDP experiment are most suitable. The social welfare experiments an example of a more advanced implementation for realistic scenarios.

- <em>IHDP experiment</em>: an analysis of the capability to exclude unfair information from predictive distributions. The experiment connects to earlier research in the field, and the data is publicly available.
- <em>Simulation experiment</em>: by making use of simulated data, this experiment explores an application of the method in which black-box models are audited on fairness.
- <em>Social welfare experiment</em>: a large scale real data experiment in the context of detecting unlawful receivers of social welfare.  

## Questions and Issues

For any questions, comments or ideas, don't hesitate to contact Rik (helwegen.rc@gmail.com)
