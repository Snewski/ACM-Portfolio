// MODEL FOR RL-LEARNING AGENT

data{
  int<lower = 1> players; //number of players
  int<lower = 1> n; //number trials 
  array[n]  int choice; //choices - integers go in, for Matching Pennies 0 or 1. Array "choice" of length n
  array[n] int feedback; //for building memory. based on opponent's choices
  real prior_alpha_m; //mean
  real<lower = 0> prior_alpha_sd;
  real prior_beta_m;
  real<lower = 0> prior_beta_sd;
}

// PARAMETERS (in case of RL) - inferred

parameters{
  real alpha_raw;//unconstrained to accommodate STAN's sampling
  real beta_raw;
}

transformed parameters { 
  real<lower=0, upper=1> alpha = inv_logit(alpha_raw); //taking the inverse-logit of alpha_raw to constrain it between 0 and 1
  real<lower=0> beta = exp(beta_raw); //exponentiating beta to ensure it's always positive
}

// MODEL BLOCK
// define priors and the likelihood here

model{
  // Priors - choosing weakly informative ones. Normal for a rate (alpha) and a coefficient (beta)
  target+= normal_lpdf(alpha_raw | 0, 1); 
  target+= normal_lpdf(beta_raw | 0, 1);
  
  // Expected values (EV) and 
  //Likelihood - how the data depend on the parameters
  vector[2] EV = rep_vector(0.5, 2); //vector with 2 elements: EV for 0 and 1. Initial state: equal P assigned to both options since there is no feedback yet
  for (t in 1:n){
    target += bernoulli_logit_lpmf(choice[t] | beta * (EV[2] - EV[1])); // choice on trial t given the difference between expected values for each option multiplied by beta
    EV[choice[t]+1] += alpha * (feedback[t] - EV[choice[t]+1]); // [t] + 1 converts 0 and 1 to 1 and 2 to accommodate STAN's indexing
  }
      //target+=bernoulli_logit_lpmf(choice | alpha, beta);
}

//GENERATED QUANTITIES BLOCK 

//N.B. for WSLS / RL / etc in which theta changes on a trial-by-trial bases, 
//we’ll need to generate predictions per each separate trial, and/or be creative 
//in terms of which predictions we want to see.

//this code is executed after sampling
generated quantities {
  // prior samples — fresh draw every iteration
  real<lower=0,upper=1> alpha_prior = inv_logit(normal_rng(prior_alpha_m, prior_alpha_sd));
  real<lower=0>         beta_prior  = exp(normal_rng(prior_beta_m, prior_beta_sd));

  array[n] int choice_prior_pred;
  array[n] int choice_posterior_pred;

  // prior predictive — fully simulated, no real data used
  vector[2] EV_prior = rep_vector(0.5, 2);
  for (t in 1:n) {
    int opponent_t = bernoulli_rng(0.5);   // simulated opponent
    choice_prior_pred[t] = bernoulli_logit_rng(beta_prior * (EV_prior[2] - EV_prior[1]));
    int fb_prior = (choice_prior_pred[t] == opponent_t) ? 1 : 0;
    EV_prior[choice_prior_pred[t]+1] += alpha_prior * (fb_prior - EV_prior[choice_prior_pred[t]+1]);
  }

  // posterior predictive — uses real feedback, inferred parameters
  vector[2] EV_post = rep_vector(0.5, 2);
  for (t in 1:n) {
    choice_posterior_pred[t] = bernoulli_logit_rng(beta * (EV_post[2] - EV_post[1]));
    EV_post[choice_posterior_pred[t]+1] += alpha * (feedback[t] - EV_post[choice_posterior_pred[t]+1]);
  }
}