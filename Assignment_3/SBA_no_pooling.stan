// Simple Bayesian Agent (SBA) — No pooling version
// Each participant is modeled independently, but with no free parameters.
// Jeffreys prior pseudo-counts: alpha0 = beta0 = 0.5.

data {
  int<lower=1> N;                 // total trials
  int<lower=1> P;                 // number of participants
  array[N] int<lower=1,upper=P> id; // participant index for each trial

  array[N] int<lower=0, upper=7> choice;    // final rating
  array[N] int<lower=0, upper=7> rating_1;  // private rating
  array[N] int<lower=0, upper=7> social;    // social rating

  int<lower=0, upper=1> run_diagnostics;
}

transformed data {
  real alpha0 = 0.5;
  real beta0  = 0.5;
}

model {
  // No parameters — pure likelihood
  vector[N] alpha_post;
  vector[N] beta_post;

  for (n in 1:N) {
    alpha_post[n] = alpha0 + rating_1[n] + social[n];
    beta_post[n]  = beta0  + (7 - rating_1[n]) + (7 - social[n]);
  }

  target += beta_binomial_lpmf(choice | 7, alpha_post, beta_post);
}

generated quantities {
  vector[N] log_lik;
  array[N] int prior_pred;
  array[N] int posterior_pred;

  if (run_diagnostics) {
    for (n in 1:N) {
      real alpha_post_n = alpha0 + rating_1[n] + social[n];
      real beta_post_n  = beta0  + (7 - rating_1[n]) + (7 - social[n]);

      log_lik[n]        = beta_binomial_lpmf(choice[n] | 7, alpha_post_n, beta_post_n);
      prior_pred[n]     = beta_binomial_rng(7, alpha0, beta0);
      posterior_pred[n] = beta_binomial_rng(7, alpha_post_n, beta_post_n);
    }
  }
}