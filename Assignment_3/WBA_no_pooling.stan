// Weighted Bayesian Agent (reparameterised) — no pooling across participants.
// Each participant j has their own rho_j and kappa_j.

data {
  int<lower=1> N;                               // total trials
  array[N] int<lower=0, upper=7> choice;        // SecondRating
  array[N] int<lower=0, upper=7> rating_1;      // FirstRating
  array[N] int<lower=0, upper=7> social;        // GroupRating

  int<lower=1> J;                               // number of participants
  array[N] int<lower=1, upper=J> id;            // participant index for each trial

  real prior_rho_alpha;
  real prior_rho_beta;
  real prior_kappa_mu;
  real<lower=0> prior_kappa_sigma;
  int<lower=0, upper=1> run_diagnostics;
}

parameters {
  array[J] real<lower=0, upper=1> rho;   // participant-specific rho_j
  array[J] real<lower=0>          kappa; // participant-specific kappa_j
}

transformed parameters {
  array[J] real<lower=0> weight_direct;
  array[J] real<lower=0> weight_social;

  for (j in 1:J) {
    weight_direct[j] = rho[j] * kappa[j];
    weight_social[j] = (1.0 - rho[j]) * kappa[j];
  }
}

model {
  // Priors: independent across participants (no pooling)
  for (j in 1:J) {
    rho[j]   ~ beta(prior_rho_alpha, prior_rho_beta);
    kappa[j] ~ lognormal(prior_kappa_mu, prior_kappa_sigma);
  }

  // Likelihood
  for (n in 1:N) {
    int j = id[n];
    real alpha_post = 0.5
      + weight_direct[j] * rating_1[n]
      + weight_social[j] * social[n];
    real beta_post  = 0.5
      + weight_direct[j] * (7 - rating_1[n])
      + weight_social[j] * (7 - social[n]);

    target += beta_binomial_lpmf(choice[n] | 7, alpha_post, beta_post);
  }
}

generated quantities {
  vector[N] log_lik;
  array[N] int prior_pred;
  array[N] int posterior_pred;
  real lprior = 0;

  // For convenience, store summed prior log-density
  for (j in 1:J) {
    lprior += beta_lpdf(rho[j] | prior_rho_alpha, prior_rho_beta)
            + lognormal_lpdf(kappa[j] | prior_kappa_mu, prior_kappa_sigma);
  }

  if (run_diagnostics) {
    // Draw one prior sample per participant for prior predictive
    array[J] real rho_prior;
    array[J] real kappa_prior;
    array[J] real wd_prior;
    array[J] real ws_prior;

    for (j in 1:J) {
      rho_prior[j]   = beta_rng(prior_rho_alpha, prior_rho_beta);
      kappa_prior[j] = lognormal_rng(prior_kappa_mu, prior_kappa_sigma);
      wd_prior[j]    = rho_prior[j] * kappa_prior[j];
      ws_prior[j]    = (1.0 - rho_prior[j]) * kappa_prior[j];
    }

    for (n in 1:N) {
      int j = id[n];

      // Posterior quantities
      real alpha_post = 0.5
        + weight_direct[j] * rating_1[n]
        + weight_social[j] * social[n];
      real beta_post  = 0.5
        + weight_direct[j] * (7 - rating_1[n])
        + weight_social[j] * (7 - social[n]);

      // Prior quantities
      real alpha_prior = 0.5
        + wd_prior[j] * rating_1[n]
        + ws_prior[j] * social[n];
      real beta_prior  = 0.5
        + wd_prior[j] * (7 - rating_1[n])
        + ws_prior[j] * (7 - social[n]);

      log_lik[n]        = beta_binomial_lpmf(choice[n] | 7, alpha_post, beta_post);
      prior_pred[n]     = beta_binomial_rng(7, alpha_prior,    beta_prior);
      posterior_pred[n] = beta_binomial_rng(7, alpha_post,     beta_post);
    }
  }
}
