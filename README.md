# ACM-Portfolio #

## Assignment 1 ##
In Assignment 1, two strategies were implemented to play the Matching Pennies game against a biased opponent, and an opponent using the Win-Stay Loose-Shift strategy. 

1) Beta Memory Strategy

The beta strategy uses the mean of a beta distribution to infer the bias of the opponent.

This strategy was implemented both deterministically and probabilistically.

2) Reinforcement Learning Strategy

The RL agent updates its belief about the probability of opponent choosing heads using the Delta Rule: it adjusts its expected value based on the prediction error between expected and observed outcomes. 

A learning rate α determines how strongly each new outcome influences the belief.

## Assignment 2 ##

In Assignment 2, the reinforcement learning (RL) model from Assignment 1 was formalized as a cognitive model and implemented in Stan.

The workflow included:

Model specification in Stan using the Delta Rule update for the expected value on each trial.

Bayesian inference to recover the learning rate (α) and inverse‑temperature (β) from simulated agent behavior.

Model quality checks, including trace plots, prior–posterior updates, and posterior predictive checks to assess convergence and fit.

Parameter recovery analyses across different trial lengths to evaluate identifiability and estimation accuracy.

## Assignment 3 ##

In Assignment 3, real‑world behavioral data were analyzed using Bayesian cognitive models of trustworthiness judgments. 

Two cognitive models were implemented in R and Stan:

Simple Bayesian Agent (SBA): assumes participants treat their own rating and the social rating as equally informative, updating a Beta prior with both sources of evidence.

Weighted Bayesian Agent (WBA): extends the SBA by allowing different weights for private and social information, enabling flexible evidence integration.

The workflow included:

Scenario design: 3 simulated scenarios were created to highlight behavioral differences between the SBA and WBA.

Model quality checks: convergence diagnostics (trace plots, divergences, R‑hat), prior–posterior updates, and posterior predictive checks were used to assess model fit and sampling behavior.

Empirical data analysis: both models were fit to a real dataset using a no‑pooling approach (each participant fit individually).

Model comparison: participant‑level ELPD differences were computed to determine which model best captured human behavior.

## Assignment 4 ##

In Assignment 4, the goal was to build and validate a formal cognitive model of categorization using data from the alien game

To capture how participants learned the category structure, a Generalized Context Model (GCM) was implemented as a similarity‑based learner. 
Rather than inferring explicit rules, the GCM assumes that participants store previously encountered aliens and judge new ones by comparing them to these stored exemplars. 

The model includes:

Feature attention weights (w): how strongly each feature contributes to similarity.

Sensitivity (c): how sharply similarity declines with feature differences.

Bias (β): a prior tendency to classify aliens as dangerous or safe.

The workflow included:

Task simulation: implemented the full alien‑game structure (5 binary features, 32 stimuli × 3 blocks) and simulated agents using the Generalized Context Model (GCM).

Scenario design: created multiple parameter scenarios (different attention‑weight profiles and sensitivity values) to illustrate how the GCM predicts different learning behaviors.

Model quality checks: prior–posterior updates, and posterior predictive checks.

Empirical data analysis: explored real participant data (dyads and individuals), computed learning curves, and fit the GCM to the participants.
