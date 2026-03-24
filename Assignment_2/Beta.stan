data {
  int<lower=1> n;
  array[n] int<lower=0, upper=1> self;
  array[n] int<lower=0, upper=1> other;
}

parameters {
  real<lower=0> alpha_init;
  real<lower=0> beta_init;
}

transformed parameters {
  array[n] real<lower=0> alpha;
  array[n] real<lower=0> beta;
  array[n] real<lower=0, upper=1> theta_mean;

  // initial values
  alpha[1] = alpha_init;
  beta[1]  = beta_init;
  theta_mean[1] = alpha[1] / (alpha[1] + beta[1]);

  // recursive updates
  for (t in 2:n) {
    alpha[t] = alpha[t-1] + other[t-1];
    beta[t]  = beta[t-1] + (1 - other[t-1]);
    theta_mean[t] = alpha[t] / (alpha[t] + beta[t]);
  }
}

model {
  // priors (change these, maybe just uninformative priors eg. just 1)
  target += exponential_lpdf(alpha_init | 1);
  target += exponential_lpdf(beta_init  | 1);

  // likelihood
  for (t in 1:n) {
    target += bernoulli_lpmf(self[t] | theta_mean[t]);
  }
}

generated quantities {
  array[n] real<lower=0, upper=1> theta_t0;
  array[n] real<lower=0, upper=1> theta_posterior;
  array[n] int prior_preds;
  array[n] int posterior_preds;

  // --- initial values ---
  theta_t0[1] = beta_rng(alpha[1], beta[1]);
  theta_posterior[1]  = beta_rng(alpha[1], beta[1]);

  prior_preds[1]     = bernoulli_rng(theta_t0[1]);
  posterior_preds[1] = bernoulli_rng(theta_posterior[1]);

  // --- recursive steps ---
  for (t in 2:n) {

    theta_t0[t] = beta_rng(alpha[1], beta[1]);
    
    // sample θ from prior and posterior (same distribution)
    theta_posterior[t]  = beta_rng(alpha[t-1], beta[t-1]);

    // Bernoulli predictions
    prior_preds[t]     = bernoulli_rng(theta_t0[t]);
    posterior_preds[t] = bernoulli_rng(theta_posterior[t]);
  }
}