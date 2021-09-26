rho_model_gamma_string = """
data {
    int j_total;
    
    // data for the initial wave 
    int N;                                                      // data length num days
    int K;                                                      // Number of mobility indices
    int j_first_wave;                                                      // Number of states in first wave
    matrix[N,j_first_wave] Reff;                                           // response
    matrix[N,K] Mob[j_first_wave];                                         // Mobility indices
    matrix[N,K] Mob_std[j_first_wave];                                     // std of mobility
    matrix[N,j_first_wave] sigma2;                                         // Variances of R_eff from previous study
    vector[N] policy;                                           // Indicators for post policy or not
    matrix[N,j_first_wave] local;                                          // local number of cases
    matrix[N,j_first_wave] imported;                                       // imported number of cases
 
    // data for the secondary wave 
    int N_sec_wave;                                             // length of VIC days
    int j_sec_wave;                                             // second wave states
    matrix[N_sec_wave,j_sec_wave] Reff_sec_wave;                // Reff for VIC in June
    matrix[N_sec_wave,K] Mob_sec_wave[j_sec_wave];              // Mob for VIC June
    matrix[N_sec_wave,K] Mob_sec_wave_std[j_sec_wave];          // std of mobility
    matrix[N_sec_wave,j_sec_wave] sigma2_sec_wave;              // variance of R_eff from previous study
    vector[N_sec_wave] policy_sec_wave;                         // micro distancing compliance
    matrix[N_sec_wave,j_sec_wave] local_sec_wave;               // local cases in VIC
    matrix[N_sec_wave,j_sec_wave] imported_sec_wave;            // imported cases in VIC

    // data for the third wave  
    int N_third_wave;                                           // length of VIC days
    int j_third_wave;                                           // third wave states
    matrix[N_third_wave,j_third_wave] Reff_third_wave;          // Reff for VIC in June
    matrix[N_third_wave,K] Mob_third_wave[j_third_wave];        // Mob for VIC June
    matrix[N_third_wave,K] Mob_third_wave_std[j_third_wave];    // std of mobility
    matrix[N_third_wave,j_third_wave] sigma2_third_wave;        // variance of R_eff from previous study
    vector[N_third_wave] policy_third_wave;                     // micro distancing compliance
    matrix[N_third_wave,j_third_wave] local_third_wave;         // local cases in VIC
    matrix[N_third_wave,j_third_wave] imported_third_wave;      // imported cases in VIC

    // data relating to mobility and microdistancing
    vector[N] count_md[j_first_wave];                                      // count of always
    vector[N] respond_md[j_first_wave];                                    // num respondants
    vector[N_sec_wave] count_md_sec_wave[j_sec_wave];           // count of always
    vector[N_sec_wave] respond_md_sec_wave[j_sec_wave];         // num respondants
    vector[N_third_wave] count_md_third_wave[j_third_wave];     // count of always
    vector[N_third_wave] respond_md_third_wave[j_third_wave];   // num respondants

    int map_to_state_index_first[j_first_wave];                     // indices of second wave to map to first
    int map_to_state_index_sec[j_sec_wave];                     // indices of second wave to map to first
    int map_to_state_index_third[j_third_wave];                 // indices of second wave to map to first
    int total_N_p_sec;                                          // total number of data in sec wave, entire state first
    int total_N_p_third;                                        // total number of data in sec wave, entire state first
    vector[N] include_in_first_wave[j_first_wave];                // dates include in first wave 
    vector[N_sec_wave] include_in_sec_wave[j_sec_wave];         // dates include in sec_wave 
    vector[N_third_wave] include_in_third_wave[j_third_wave];   // dates include in sec_wave 
    int pos_starts_sec[j_sec_wave];                             // starting positions for each state in the second wave
    int pos_starts_third[j_third_wave];                         // starting positions for each state in the third wave 
    
    int is_ACT[j_third_wave];                                   // indicator vector of which state is NSW in the third wave
    int is_NSW[j_third_wave];                                   // indicator vector of which state is NSW in the third wave
    int days_late_third_wave;                                   // number of days that ACT's outbreak started after NSW and VIC 

    int decay_start_date_third[2];
    vector[N_third_wave] vaccine_effect_data[j_third_wave];     //vaccination data

}
parameters {
    vector[K] bet;                                              // coefficients
    real<lower=0> R_I;                                          // base level imports,
    real<lower=0> R_L;                                          // base level local
    vector<lower=0>[j_total] R_Li;                                    // state level estimates
    real<lower=0> sig;                                          // state level variance
    real<lower=0> theta_md;                                     // md weighting
    matrix<lower=0,upper=1>[N,j_first_wave] prop_md;                       // proportion who are md'ing
    vector<lower=0,upper=1>[total_N_p_sec] prop_md_sec_wave;
    vector<lower=0,upper=1>[total_N_p_third] prop_md_third_wave;
    matrix<lower=0,upper=1>[N,j_first_wave] brho;                          // estimate of proportion of imported cases
    vector<lower=0,upper=1>[total_N_p_sec] brho_sec_wave;       // estimate of proportion of imported cases
    vector<lower=0,upper=1>[total_N_p_third] brho_third_wave;   // estimate of proportion of imported cases

    // voc and vaccine effects
    real<lower=0> voc_effect_sec_wave;
    real<lower=0> voc_effect_third_wave;
    real<lower=0,upper=1> eta_NSW;                              // array of adjustment factor for each third wave state
    real<lower=0,upper=1> eta_other;                            // array of adjustment factor for each third wave state
    real<lower=0> r_NSW;                                        // parameter for decay to heterogeneity
    real<lower=0> r_other;                                      // parameter for decay to heterogeneity

}
transformed parameters {
    matrix<lower=0>[N,j_first_wave] mu_hat;
    vector<lower=0>[total_N_p_sec] mu_hat_sec_wave;
    vector<lower=0>[total_N_p_third] mu_hat_third_wave;
    matrix<lower=0>[N,j_first_wave] md;                                    // micro distancing
    vector<lower=0>[total_N_p_sec] md_sec_wave;
    vector<lower=0>[total_N_p_third] md_third_wave;

    for (i in 1:j_first_wave) {
        for (n in 1:N){
            if (include_in_first_wave[i][n]==1) {
                md[n,i] = pow(1+theta_md , -1*prop_md[n,i]);
                //mean estimate
                mu_hat[n,i] = brho[n,i]*R_I + (1-brho[n,i])*2*R_Li[map_to_state_index_first[i]]*(
                (1-policy[n]) + md[n,i]*policy[n] )*inv_logit(Mob[i][n,:]*(bet)); 
            }
        }
    }
    for (i in 1:j_sec_wave){
        // define these within the scope of the loop only
        int pos;
        if (i==1){
            pos=1;
        }
        else {
            //Add 1 to get to start of new group, not end of old group
            pos =pos_starts_sec[i-1]+1;
            }
        for (n in 1:N_sec_wave){
            if (include_in_sec_wave[i][n]==1){        
                md_sec_wave[pos] = pow(1+theta_md ,-1*prop_md_sec_wave[pos]);
                if (map_to_state_index_sec[i] == 5) {       // include VIC separately
                    mu_hat_sec_wave[pos] = brho_sec_wave[pos]*R_I + (1-brho_sec_wave[pos])*(2*R_Li[
                    map_to_state_index_sec[i]
                    ])*(
                    (1-policy_sec_wave[n]) + md_sec_wave[pos]*policy_sec_wave[n] )*inv_logit(
                    Mob_sec_wave[i][n,:]*(bet))*voc_effect_sec_wave; //mean estimate
                }
                else {
                    mu_hat_sec_wave[pos] = brho_sec_wave[pos]*R_I + (1-brho_sec_wave[pos])*2*R_Li[
                    map_to_state_index_sec[i]
                    ]*(
                    (1-policy_sec_wave[n]) + md_sec_wave[pos]*policy_sec_wave[n] )*inv_logit(
                    Mob_sec_wave[i][n,:]*(bet))*voc_effect_sec_wave; //mean estimate
                }
                pos += 1;
            }
        }
    }

    for (i in 1:j_third_wave){
        // define these within the scope of the loop only
        int pos;
        real vacc_effect_tot;
        real eta;
        real eta_tmp;
        real r;
        real decay_in_heterogeneity;
        real decay_start_date_adjusted;
        
        if (i==1){
            pos=1;
        } else {
            //Add 1 to get to start of new group, not end of old group
            pos = pos_starts_third[i-1]+1;
        }

        // apply different heterogeneity effects depending on whether we are looking at NSW or not
        if (is_NSW[i] == 1){
            eta = eta_NSW;
            r = r_NSW;
        } else {
            eta = eta_other;
            r = r_other;
        }
        
        // pick number of days after third start date, only difference is that ACT's third wave started 
        // slightly later than NSW and VIC. This offset ensures that the exponent on the decay term is 
        // appropriately sized. 
        if (is_ACT[i] == 1){ 
            decay_start_date_adjusted = decay_start_date_third[1];
        } else {
            decay_start_date_adjusted = decay_start_date_third[2];
        }

        for (n in 1:N_third_wave){
            if (include_in_third_wave[i][n]==1){
                md_third_wave[pos] = pow(1+theta_md ,-1*prop_md_third_wave[pos]);                

                // applying the return to homogeneity in vaccination effect 
                if (n < decay_start_date_adjusted){
                    decay_in_heterogeneity = 1;
                } else{
                    decay_in_heterogeneity = exp(-r*(n-decay_start_date_adjusted));
                }
                
                eta_tmp = eta*decay_in_heterogeneity;

                // total vaccination effect has the form of a mixture model which captures heterogeneity in the 
                // vaccination effect around the 20th of August 
                vacc_effect_tot = eta_tmp + (1-eta_tmp) * vaccine_effect_data[i][n];
                
                mu_hat_third_wave[pos] = brho_third_wave[pos]*R_I + 
                    (1-brho_third_wave[pos])*2*R_Li[map_to_state_index_third[i]]*
                    ((1-policy_third_wave[n]) + md_third_wave[pos]*policy_third_wave[n])*
                    inv_logit(Mob_third_wave[i][n,:]*(bet))*voc_effect_third_wave*vacc_effect_tot;
                pos += 1;
            }
        }
    }
}
model {
    int pos2;

    bet ~ normal(0,1.0);
    theta_md ~ lognormal(0,0.5);

    // note gamma parametrisation is Gamma(alpha,beta) => mean = alpha/beta 
    voc_effect_sec_wave ~ gamma(1.3*1.3/0.05, 1.3/0.05);
    voc_effect_third_wave ~ gamma(2.6*2.6/0.05, 2.6/0.05);
    
    // assume a hierarchical structure on the vaccine effect 
    eta_NSW ~ beta(2, 7);           // mean of 2/9
    eta_other ~ beta(2, 7);         // mean of 2/9

    // want it to have mean 0.16 => log-mean is log(0.16)
    r_NSW ~ lognormal(log(0.16),0.1);        // r is lognormally distributed such that the mean is 28 days 
    r_other ~ lognormal(log(0.16),0.1);        // r is lognormally distributed such that the mean is 28 days 

    R_L ~ gamma(1.8*1.8/0.01,1.8/0.01); //hyper-prior
    R_I ~ gamma(0.5*0.5/0.2,0.5/0.2);
    sig ~ exponential(50); //mean is 1/50=0.02
    R_Li ~ gamma(R_L*R_L/sig,R_L/sig); //partial pooling of state level estimates

    for (i in 1:j_first_wave) {
        for (n in 1:N){
            prop_md[n,i] ~ beta(1 + count_md[i][n], 1 + respond_md[i][n] - count_md[i][n]);
            brho[n,i] ~ beta( 1+ imported[n,i], 1+ local[n,i]); //ratio imported/ (imported + local)
            mu_hat[n,i] ~ gamma( Reff[n,i]*Reff[n,i]/(sigma2[n,i]), Reff[n,i]/sigma2[n,i]); //Stan uses shape/inverse scale
        }
    }
    
    for (i in 1:j_sec_wave){
        if (i==1){
            pos2=1;
        }
        else {
            //Add 1 to get to start of new group, not end of old group
            pos2 =pos_starts_sec[i-1]+1; 
            }
        for (n in 1:N_sec_wave){
            if (include_in_sec_wave[i][n]==1){
                prop_md_sec_wave[pos2] ~ beta(1 + count_md_sec_wave[i][n], 1+ respond_md_sec_wave[i][n] - count_md_sec_wave[i][n]);
                brho_sec_wave[pos2] ~ beta( 1+ imported_sec_wave[n,i], 1+ local_sec_wave[n,i]); //ratio imported/ (imported + local)
                mu_hat_sec_wave[pos2] ~ gamma( Reff_sec_wave[n,i]*Reff_sec_wave[n,i]/(sigma2_sec_wave[n,i]), Reff_sec_wave[n,i]/sigma2_sec_wave[n,i]);
                pos2+=1;
            }
        }
    }

    for (i in 1:j_third_wave){
        if (i==1){
            pos2=1;
        } else {
            //Add 1 to get to start of new group, not end of old group
            pos2 =pos_starts_third[i-1]+1; 
        }
        for (n in 1:N_third_wave){
            if (include_in_third_wave[i][n]==1){
                prop_md_third_wave[pos2] ~ beta(1 + count_md_third_wave[i][n], 1+ respond_md_third_wave[i][n] - count_md_third_wave[i][n]);
                brho_third_wave[pos2] ~ beta( 1+ imported_third_wave[n,i], 1+ local_third_wave[n,i]); //ratio imported/ (imported + local)
                mu_hat_third_wave[pos2] ~ gamma(Reff_third_wave[n,i]*Reff_third_wave[n,i]/(sigma2_third_wave[n,i]), Reff_third_wave[n,i]/sigma2_third_wave[n,i]);
                pos2+=1;
            }
        }
    }
}

// model {
//     int pos2;

//     target += normal_lpdf(bet | 0, 1.0);
//     target += lognormal_lpdf(theta_md | 0, 0.5);
//     target += gamma_lpdf(voc_effect_sec_wave | 1.3*1.3/0.05, 1.3/0.05);
//     target += gamma_lpdf(voc_effect_third_wave | 2.9*2.9/0.05, 2.9/0.05);
//     target += beta_lpdf(eta_NSW | 2, 7); 
//     target += beta_lpdf(eta_other | 2, 7); 
//     target += lognormal_lpdf(r_NSW | log(0.16), 0.1);
//     target += lognormal_lpdf(r_other | log(0.16), 0.1);
//     target += gamma_lpdf(R_L | 1.8*1.8/0.01,1.8/0.01);
//     target += gamma_lpdf(R_I | 0.5*0.5/0.2,0.5/0.2);
//     target += exponential_lpdf(sig | 50);
//     target += gamma_lpdf(R_Li | R_L*R_L/sig, R_L/sig);

//     for (i in 1:j_first_wave) {
//         for (n in 1:N){
//             target += beta_lpdf(prop_md[n,i] | 1 + count_md[i][n], 1 + respond_md[i][n] - count_md[i][n]);
//             target += beta_lpdf(brho[n,i] | 1+ imported[n,i], 1+ local[n,i]); //ratio imported/ (imported + local)
//             target += gamma_lpdf(mu_hat[n,i] | Reff[n,i]*Reff[n,i]/(sigma2[n,i]), Reff[n,i]/sigma2[n,i]); //Stan uses shape/inverse scale
//         }
//     }
    
//     for (i in 1:j_sec_wave){
//         if (i==1){
//             pos2=1;
//         }
//         else {
//             //Add 1 to get to start of new group, not end of old group
//             pos2 =pos_starts_sec[i-1]+1; 
//             }
//         for (n in 1:N_sec_wave){
//             if (include_in_sec_wave[i][n]==1){
//                 target += beta_lpdf(prop_md_sec_wave[pos2] | 1 + count_md_sec_wave[i][n], 1+ respond_md_sec_wave[i][n] - count_md_sec_wave[i][n]);
//                 target += beta_lpdf(brho_sec_wave[pos2] | 1+ imported_sec_wave[n,i], 1+ local_sec_wave[n,i]); //ratio imported/ (imported + local)
//                 target += gamma_lpdf(mu_hat_sec_wave[pos2] | Reff_sec_wave[n,i]*Reff_sec_wave[n,i]/(sigma2_sec_wave[n,i]), Reff_sec_wave[n,i]/sigma2_sec_wave[n,i]);
//                 pos2+=1;
//             }
//         }
//     }

//     for (i in 1:j_third_wave){
//         if (i==1){
//             pos2=1;
//         } else {
//             //Add 1 to get to start of new group, not end of old group
//             pos2 =pos_starts_third[i-1]+1; 
//         }
//         for (n in 1:N_third_wave){
//             if (include_in_third_wave[i][n]==1){
//                 target += beta_lpdf(prop_md_third_wave[pos2] | 1 + count_md_third_wave[i][n], 1+ respond_md_third_wave[i][n] - count_md_third_wave[i][n]);
//                 target += beta_lpdf(brho_third_wave[pos2] | 1+ imported_third_wave[n,i], 1+ local_third_wave[n,i]); //ratio imported/ (imported + local)
//                 target += gamma_lpdf(mu_hat_third_wave[pos2] | Reff_third_wave[n,i]*Reff_third_wave[n,i]/(sigma2_third_wave[n,i]), Reff_third_wave[n,i]/sigma2_third_wave[n,i]);
//                 pos2+=1;
//             }
//         }
//     }
// }
"""
