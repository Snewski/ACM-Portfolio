//
// This Stan program defines a simple model, with a
// vector of values 'y' modeled as normally distributed
// with mean 'mu' and standard deviation 'sigma'.
//
// Learn more about model development with Stan at:
//
//    http://mc-stan.org/users/interfaces/rstan.html
//    https://github.com/stan-dev/rstan/wiki/RStan-Getting-Started
//

// // The input data is a vector 'y' of length 'N'.
// data {
//   int<lower=0> N;
//   vector[N] y;
// }
// 
// // The parameters accepted by the model. Our model
// // accepts two parameters 'mu' and 'sigma'.
// parameters {
//   real mu;
//   real<lower=0> sigma;
// }
// 
// // The model to be estimated. We model the output
// // 'y' to be normally distributed with mean 'mu'
// // and standard deviation 'sigma'.
// model {
//   y ~ normal(mu, sigma);
// }

// MODEL FOR RL-LEARNING AGENT

data{
  int<lower = 1> players; //number of participants
  int<lower = 1> n; //n trials - probably not necessary in here
  array[n]  int choice; //choices - integers go in. Array "choice" of length n
  array[n] int feedback; //for building memory - again check if this should be defined here or in R
  real prior_alpha_m; //mean
  real<lower = 0> prior_alpha_sd;
  real prior_beta_m;
  real<lower = 0> prior_beta_sd;
}

transformed data {
  // does the RL model requrie anything here?
}

// PARAMETERS (in case of RL) - inferred

parameters{
  real alpha_raw;//unconstrained for now
  real beta_raw;
  //real<lower = 0, upper = 1> alpha_raw;
  //real<lower = 0, upper = 20> beta_raw; //if we have multiple participants, it should be a vector not a real number. upper bound is freely decided given knowledge (technically it could be infinite but unlikely)
}

transformed parameters { //again check this, figure out logit thing
  real<lower=0, upper=1> alpha = inv_logit(alpha_raw);
  real<lower=0> beta = exp(beta_raw);
}

// MODEL BLOCK
// define priors and the likelihood here

model{
  // Priors
  target+= normal_lpdf(alpha_raw | prior_alpha_m, prior_alpha_sd); //should these be on the logit?
  target+= normal_lpdf(beta_raw | prior_beta_m, prior_beta_sd);
  
  // Expected values (EV) and 
  //Likelihood - how the data depend on the parameters
  vector[2] EV = rep_vector(0.5, 2);
  for (t in 1:n){
    target += bernoulli_logit_lpmf(choice[t] | beta * (EV[2] - EV[1]));
    EV[choice[t]+1] += alpha * (feedback[t] - EV[choice[t]+1]);
  }
      //target+=bernoulli_logit_lpmf(choice | alpha, beta);
}

//GENERATED QUANTITIES BLOCK 

//N.B. for WSLS / RL / etc in which theta changes on a trial-by-trial bases, 
//we’ll need to generate predictions per each separate trial, and/or be creative 
//in terms of which predictions we want to see.

//this code is executed after sampling
generated quantities {
  // saving priors
  real<lower = 0, upper = 1> alpha_prior = inv_logit(normal_rng(prior_alpha_m, prior_alpha_sd));
  real<lower = 0> beta_prior = exp(normal_rng(prior_beta_m, prior_beta_sd));
  
  array[n] int choice_prior_pred;
  array[n] int choice_posterior_pred;
  
  vector[2] EV_prior = rep_vector(0.5, 2);
  for (t in 1:n) {
    choice_prior_pred[t] = bernoulli_logit_rng(beta_prior * (EV_prior[2] - EV_prior[1]));
    EV_prior[choice_prior_pred[t]+1] += alpha_prior * (feedback[t] - EV_prior[choice_prior_pred[t]+1]);
  }
  
  vector[2] EV_post = rep_vector(0.5, 2);
  for (t in 1:n){
    choice_posterior_pred[t] = bernoulli_logit_rng(beta * (EV_post[2] - EV_post[1]));
    EV_post[choice_posterior_pred[t]+1] += alpha * (feedback[t] - EV_post[choice_posterior_pred[t]+1]);
  }

}
