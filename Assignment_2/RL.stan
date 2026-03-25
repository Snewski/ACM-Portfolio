data {
  int<lower=1> n;                   // number of trials
  array[n] int<lower=0, upper=1> Self;     // RL agent choices
  array[n] int<lower=0, upper=1> Other;    // WSLS agent choices
}

parameters {
  real alpha_raw;                   // unconstrained learning rate
  real beta_raw;                    // unconstrained inverse temperature
}

transformed parameters {
  real<lower=0, upper=1> alpha = inv_logit(alpha_raw);
  real<lower=0>   beta  = exp(beta_raw);

  array[n] real<lower=0, upper=1> V;       // latent value trajectory
  array[n] real<lower=0, upper=1> p;       // choice probabilities

  // initial values
  V[1] = 0.5;
  p[1] = inv_logit(beta * (V[1] - 0.5));

  // RL update uses Other[t-1]
  for (t in 2:n) {
    V[t] = V[t-1] + alpha * (Other[t-1] - V[t-1]);
    p[t] = inv_logit(beta * (V[t] - 0.5));
  }
}

model {
  // priors
  target += normal_lpdf(alpha_raw | 0, 1);
  target += normal_lpdf(beta_raw  | 0, 1);

  // likelihood
  for (t in 1:n) {
    target += bernoulli_lpmf(Self[t] | p[t]);
  }
}

generated quantities {
  
  // prior draws (for prior–posterior comparison)
  real<lower=0, upper=1> alpha_prior = inv_logit(normal_rng(0, 1));
  real<lower=0>   beta_prior  = exp(normal_rng(0, 1));

  
  array[n] int<lower=0, upper=1> Self_rep;   // posterior predictive

  for (t in 1:n) {
    Self_rep[t] = bernoulli_rng(p[t]);
  }
}