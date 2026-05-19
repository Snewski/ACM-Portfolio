
// Simple Bayesian Agent (SBA) — Face Trustworthiness Rating.
// No free parameters — evidence is counted at face value.
// Jeffreys prior pseudo-counts: alpha0 = beta0 = 0.5.
// Ratings are on a 0-7 scale (original 1-8 shifted by -1 in R).
data {
  int<lower=1> N;
  array[N] int<lower=0, upper=7> choice;  // agent's final rating (0-7)
  array[N] int<lower=0, upper=7> rating_1; // agent's own initial rating (0-7)
  array[N] int<lower=0, upper=7> social;   // group's rating (0-7)
  int<lower=0, upper=1> run_diagnostics;
}
transformed data {
  real alpha0 = 0.5;
  real beta0  = 0.5;
}
model {
  // Vectorized likelihood — fixed weights = 1
  vector[N] alpha_post = alpha0 + to_vector(rating_1) + to_vector(social);
  vector[N] beta_post  = beta0
                         + (7 - to_vector(rating_1))
                         + (7 - to_vector(social));
  target += beta_binomial_lpmf(choice | 7, alpha_post, beta_post);
}
generated quantities {
  vector[N] log_lik;
  array[N] int prior_pred;
  array[N] int posterior_pred;

  if (run_diagnostics) {
    for (n in 1:N) {
      real alpha_post = alpha0 + rating_1[n] + social[n];
      real beta_post  = beta0  + (7 - rating_1[n]) + (7 - social[n]);

      log_lik[n]        = beta_binomial_lpmf(choice[n] | 7, alpha_post, beta_post);
      prior_pred[n]     = beta_binomial_rng(7, alpha0, beta0);
      posterior_pred[n] = beta_binomial_rng(7, alpha_post, beta_post);
    }
  }
}

