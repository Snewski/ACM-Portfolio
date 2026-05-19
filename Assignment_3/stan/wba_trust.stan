
// Weighted Bayesian Agent (reparameterised) — Face Trustworthiness Rating.
// rho in (0,1): relative weight of direct vs social evidence.
// kappa > 0: total evidence scaling (w_d + w_s).
// Ratings are on a 0-7 scale (original 1-8 shifted by -1 in R).
data {
  int<lower=1> N;
  array[N] int<lower=0, upper=7> choice;   // agent's final rating (0-7)
  array[N] int<lower=0, upper=7> rating_1; // agent's own initial rating (0-7)
  array[N] int<lower=0, upper=7> social;   // group's rating (0-7)

  real prior_rho_alpha;
  real prior_rho_beta;
  real prior_kappa_mu;
  real<lower=0> prior_kappa_sigma;
  int<lower=0, upper=1> run_diagnostics;
}
parameters {
  real<lower=0, upper=1> rho;   // relative weight: w_d / (w_d + w_s)
  real<lower=0>          kappa; // total weight: w_d + w_s
}
transformed parameters {
  real<lower=0> weight_direct = rho * kappa;
  real<lower=0> weight_social = (1.0 - rho) * kappa;
}
model {
  // rho: weakly centred on equal weighting
  target += beta_lpdf(rho | prior_rho_alpha, prior_rho_beta);
  // kappa: lognormal centered on 2 (SBA equivalent when rho = 0.5)
  target += lognormal_lpdf(kappa | prior_kappa_mu, prior_kappa_sigma);

  // Vectorized likelihood
  vector[N] alpha_post = 0.5 + weight_direct * to_vector(rating_1)
                             + weight_social * to_vector(social);
  vector[N] beta_post  = 0.5 + weight_direct * (7 - to_vector(rating_1))
                             + weight_social * (7 - to_vector(social));
  target += beta_binomial_lpmf(choice | 7, alpha_post, beta_post);
}
generated quantities {
  vector[N] log_lik;
  array[N] int prior_pred;
  array[N] int posterior_pred;
  real lprior = beta_lpdf(rho | prior_rho_alpha, prior_rho_beta) +
                lognormal_lpdf(kappa | prior_kappa_mu, prior_kappa_sigma);

  // Draw from priors for prior predictive
  real rho_prior   = beta_rng(prior_rho_alpha, prior_rho_beta);
  real kappa_prior = lognormal_rng(prior_kappa_mu, prior_kappa_sigma);
  real wd_prior    = rho_prior * kappa_prior;
  real ws_prior    = (1.0 - rho_prior) * kappa_prior;

  if (run_diagnostics) {
    for (n in 1:N) {
      // Posterior quantities
      real alpha_post = 0.5
        + weight_direct * rating_1[n]
        + weight_social * social[n];
      real beta_post  = 0.5
        + weight_direct * (7 - rating_1[n])
        + weight_social * (7 - social[n]);

      // Prior quantities
      real alpha_prior = 0.5
        + wd_prior * rating_1[n]
        + ws_prior * social[n];
      real beta_prior  = 0.5
        + wd_prior * (7 - rating_1[n])
        + ws_prior * (7 - social[n]);

      log_lik[n]        = beta_binomial_lpmf(choice[n] | 7, alpha_post, beta_post);
      prior_pred[n]     = beta_binomial_rng(7, alpha_prior, beta_prior);
      posterior_pred[n] = beta_binomial_rng(7, alpha_post,  beta_post);
    }
  }
}

