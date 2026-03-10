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
  // Should change this to include alpha & beta.
  array[n] real<lower=0, upper=1> theta_prior;
  array[n] real<lower=0, upper=1> theta_posterior;
  array[n] real<lower=0, upper=1> prior_preds;
  array[n] real<lower=0, upper=1> posterior_preds;

  
  // Should change this to utilize beta_rng()
  for (t in 1:n) {

    // prior mean before observing trial t
    theta_prior[t] = alpha[t] / (alpha[t] + beta[t]);

    // posterior mean after observing trial t
    if (t < n) {
      real a_post = alpha[t] + other[t];
      real b_post = beta[t] + (1 - other[t]);
      theta_posterior[t] = a_post / (a_post + b_post);
    } else {
      // last trial: posterior = prior (no future update)
      theta_posterior[t] = theta_prior[t];
    }

    // prior predictive sample
    prior_preds[t] = bernoulli_rng(theta_prior[t]);

    // posterior predictive sample
    posterior_preds[t] = bernoulli_rng(theta_posterior[t]);
  }
}