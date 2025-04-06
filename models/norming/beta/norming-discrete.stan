functions {
  // the norming model likelihood:
  real likelihood_lpdf(
		       real y,
		       real world,
		       real eta,
		       real phi
		       ) {
    real mu_1 = inv_logit(eta);

    real alpha = mu_1 * phi;
    real beta = (1 - mu_1) * phi;
    
    return log_mix(
		   world,
		   beta_lpdf(y | alpha, beta),
		   beta_lpdf(y | beta, alpha)
		   );
  }
}

data {
  // the Degen and Tonhauser (2021) norming experiment:
  int<lower=1> N_context;	      // number of contexts
  int<lower=1> N_participant;	      // number of participants
  int<lower=1> N_data;		      // number of data points
  vector<lower=0, upper=1>[N_data] y; // response (between 0 and 1)
  array[N_data] int<lower=1, upper=N_context> context; // map from data points to contexts
  array[N_data] int<lower=1, upper=N_participant> participant; // map from data points to participants
}

parameters {
  // 
  // FIXED EFFECTS
  // 
  
  // contexts:
  vector[N_context] z_omega;   // by-context z-scores for the log-odds certainty
  vector<lower=0>[N_context] sigma_omega; // by-context standard deviations for the log-odds certainty


  // 
  // RANDOM EFFECTS
  // 
  
  // by-participant random intercepts for the log-odds certainty:
  real<lower=0> sigma_epsilon_omega;	 // global scaling factor
  vector[N_participant] z_epsilon_omega; // by-participant z-scores

  // likelihood parameters:
  real<lower=0> eta;		// absolute value of the 0/1 component
  real<lower=0> phi;		// beta sample size
}

transformed parameters {
  vector[N_context] omega;	// log-odds certainty
  vector[N_participant] epsilon_omega; // by-participant intercepts for the log-odds certainty
  vector<lower=0, upper=1>[N_data] w; // certainty with participant intercepts added
  vector<lower=0, upper=1>[N_data] y1; // transformed data
  
  // 
  // DEFINITIONS
  //
  
  // non-centered parameterization of the log-odds certainty:
  for (i in 1:N_context) {
    omega[i] = sigma_omega[i] * z_omega[i];
  }

  // non-centered parameteriziation of the participant random intercepts:
  epsilon_omega = sigma_epsilon_omega * z_epsilon_omega;

  // latent parameters before jittering is added:
  for (i in 1:N_data) {
    w[i] = inv_logit(omega[context[i]] + epsilon_omega[participant[i]]);
    y1[i] = (y[i] * (N_data - 1) + 0.5) / N_data;
  }
}

model {
  //
  // FIXED EFFECTS
  // 
  
  // contexts:
  sigma_omega ~ exponential(1);
  z_omega ~ std_normal;

  
  //
  // RANDOM EFFECTS
  //
  
  // by-participant random intercepts:
  sigma_epsilon_omega ~ exponential(1);
  z_epsilon_omega ~ std_normal;


  //
  // LIKELIHOOD
  // 

  // parameters:
  eta ~ normal(1.5, 1);
  phi ~ exponential(0.1);
  
  // definition:
  for (i in 1:N_data) {
    if (y1[i] > 0 && y1[i] < 1)
      target += likelihood_lpdf(
				y1[i] |
				w[i],
				eta,
				phi
				);
    else
      target += negative_infinity();
  }
}

generated quantities {
  vector[N_data] ll; // log-likelihoods (needed for WAIC/PSIS calculations)
  
  // definition:
  for (i in 1:N_data) {
    if (y1[i] > 0 && y1[i] < 1)
      ll[i] = likelihood_lpdf(
			      y1[i] |
			      w[i],
			      eta,
			      phi
			      );
    else
      ll[i] = negative_infinity();
  }
}
