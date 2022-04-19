/*
A stan model that incorporates the effect of waning infection acquired immunity on the TP. 
*/

functions {
    real sigmoid(int t, real tau, real r, real m0, real m1) {
        /*
        Calculate a translated and adjusted sigmoid function that approximates the 
        transition from an initial proportion m0 to a final proportion m1. The parameter 
        tau is the inflection point in the curve and r is the rate of increase from m0 
        to m1. The parameter t is of type int upon calling the function, but gets cast to 
        real.
        */
        real y;
        // convert t to real 
        real t_real = 1.0 * t;
        y = m0 + (m1 - m0) / (1 + exp(-r * (t_real - tau)));
       
        return y; 
    }
}

data {
    
    // overall number of states in model 
    int j_total;
    
    // first wave data 
    int N_first;
    int K;  // number of mobility coefficients 
    int j_first;
    matrix[N_first,j_first] Reff;
    array[j_first] matrix[N_first,K] mob;
    array[j_first] matrix[N_first,K] mob_std;
    matrix[N_first,j_first] sigma2;
    vector[N_first] policy;
    matrix[N_first,j_first] local;
    matrix[N_first,j_first] imported;
    
    // second wave data 
    int N_sec;
    int j_sec;
    matrix[N_sec,j_sec] Reff_sec;
    array[j_sec] matrix[N_sec,K] mob_sec;
    array[j_sec] matrix[N_sec,K] mob_sec_std;
    matrix[N_sec,j_sec] sigma2_sec;
    vector[N_sec] policy_sec;
    matrix[N_sec,j_sec] local_sec;
    matrix[N_sec,j_sec] imported_sec;
    array[N_sec] int apply_alpha_sec;
    
    // third wave data 
    int N_third;
    int j_third;
    matrix[N_third,j_third] Reff_third;
    matrix[N_third,j_third] Reff_omicron;
    array[j_third] matrix[N_third,K] mob_third;
    array[j_third] matrix[N_third,K] mob_third_std;
    matrix[N_third,j_third] sigma2_third;
    matrix[N_third,j_third] sigma2_omicron;
    vector[N_third] policy_third;
    matrix[N_third,j_third] local_third;
    matrix[N_third,j_third] imported_third;
    
    // micro data
    array[j_first] vector[N_first] count_md;
    array[j_first] vector[N_first] respond_md;
    array[j_sec] vector[N_sec] count_md_sec;
    array[j_sec] vector[N_sec] respond_md_sec;
    array[j_third] vector[N_third] count_md_third;
    array[j_third] vector[N_third] respond_md_third;
    
    // masks data 
    array[j_first] vector[N_first] count_masks;
    array[j_first] vector[N_first] respond_masks;
    array[j_sec] vector[N_sec] count_masks_sec;
    array[j_sec] vector[N_sec] respond_masks_sec;
    array[j_third] vector[N_third] count_masks_third;
    array[j_third] vector[N_third] respond_masks_third;
    
    // vectors that map to the correct indices based on j_total 
    array[j_first] int map_to_state_index_first;
    array[j_sec] int map_to_state_index_sec;
    array[j_third] int map_to_state_index_third;
    
    // ints for moving through the total parameter vectors in the second or 
    // third waves
    int total_N_p_sec;
    int total_N_p_third;
    
    // bool arrays for when to include data 
    array[j_first] vector[N_first] include_in_first;
    array[j_sec] vector[N_sec] include_in_sec;
    array[j_third] vector[N_third] include_in_third;
    
    // this is used to index starting points in include_in_XX 
    array[j_sec] int pos_starts_sec;
    array[j_third] int pos_starts_third;
    // vax data for each strain
    array[j_third] vector[N_third] ve_delta_data;
    // omicron ve time series is the same length as delta but we fit to the 
    // appropriate range
    array[j_third] vector[N_third] ve_omicron_data;     
    // data for handling omicron 
    int omicron_start_day;
    int omicron_only_day;
    array[j_third] vector[N_third] include_in_omicron;
    int total_N_p_third_omicron;
    array[j_third] int pos_starts_third_omicron;
    
    int tau_vax_block_size; 
    int total_N_p_third_blocks;
    array[j_third] int pos_starts_third_blocks;
    int total_N_p_third_omicron_blocks;
    array[j_third] int pos_starts_third_omicron_blocks;
    
    array[j_total] int pop_size_array;
    int heterogeneity_start_day;
    
    // assumed CAR over the various periods for each jurisdiction
    matrix[N_third,j_third] p_detect; 
    
}

transformed data {
    
    // for now we just calculate the cumulative number of cases in the third wave 
    // as pre third wave cases were negligible
    matrix[N_third,j_third] CA_scaling_factor = 1.0 ./ p_detect;
    vector[N_third] CA_scaled_local_third;
    vector[N_third] local_third_cum;
    
    // to be used as the accurate number of cases when determining the import fraction
    matrix[N_third,j_third] prop_inf_14;
    matrix[N_third,j_third] prop_inf_28;
    matrix[N_third,j_third] prop_inf_42;
    matrix[N_third,j_third] prop_inf_72;
    matrix[N_third,j_third] prop_inf_102;
    matrix[N_third,j_third] prop_inf_132;
    
    // shape and scale for the likelihood in each wave
    matrix[N_first,j_first] a_mu_hat;
    matrix[N_first,j_first] b_mu_hat;
    matrix[N_sec,j_sec] a_mu_hat_sec;
    matrix[N_sec,j_sec] b_mu_hat_sec;
    matrix[N_third,j_third] a_mu_hat_third;
    matrix[N_third,j_third] b_mu_hat_third;
    matrix[N_third,j_third] a_mu_hat_omicron;
    matrix[N_third,j_third] b_mu_hat_omicron;
    
    // temp variables for calculating the numerator and denominator in prop_inf 
    real num;
    real denom; 
    int idx_start;
    int idx_end;
    
    for (i in 1:j_third) {
        // scale up the cases by the assumed CAR
        CA_scaled_local_third = ceil(local_third[:,i] .* CA_scaling_factor[:,i]);
        
        denom = pop_size_array[map_to_state_index_third[i]];
        
        for (n in 1:N_third) {
            idx_start = max(n - 13, 1);
            idx_end = min(n, N_third);
            num = sum(CA_scaled_local_third[idx_start:idx_end]);
            prop_inf_14[n,i] = fmin(num / denom, 1.0);
            
            if (n > 14) {
                idx_start = max(n - 27, 1);
                idx_end = min(n - 14, N_third);
                num = sum(CA_scaled_local_third[idx_start:idx_end]);
                prop_inf_28[n,i] = fmin(num / denom, 1.0);
            } else {
                prop_inf_28[n,i] = 0.0;
            }
            
            if (n > 28) {
                idx_start = max(n - 41, 1);
                idx_end = min(n - 28, N_third);
                num = sum(CA_scaled_local_third[idx_start:idx_end]);
                prop_inf_42[n,i] = fmin(num / denom, 1.0);
            } else {
                prop_inf_42[n,i] = 0.0;
            }
            
            if (n > 42) {
                idx_start = max(n - 71, 1);
                idx_end = min(n - 42, N_third);
                num = sum(CA_scaled_local_third[idx_start:idx_end]);
                prop_inf_72[n,i] = fmin(num / denom, 1.0);
            } else {
                prop_inf_72[n,i] = 0.0;
            }
            
            if (n > 72) {
                idx_start = max(n - 101, 1);
                idx_end = min(n - 72, N_third);
                num = sum(CA_scaled_local_third[idx_start:idx_end]);
                prop_inf_102[n,i] = fmin(num / denom, 1.0);
            } else {
                prop_inf_102[n,i] = 0.0;
            }
            
            if (n > 102) {
                idx_start = max(n - 131, 1);
                idx_end = min(n - 102, N_third);
                num = sum(CA_scaled_local_third[idx_start:idx_end]);
                prop_inf_132[n,i] = fmin(num / denom, 1.0);
            } else {
                prop_inf_132[n,i] = 0.0;
            }
        }
    }

    // compute the shape and scale for the likelihood 
    for (i in 1:j_first) {
        a_mu_hat[:,i] = square(Reff[:,i]) ./ sigma2[:,i];
        b_mu_hat[:,i] = Reff[:,i] ./ sigma2[:,i];
    }
    
    for (i in 1:j_sec) {
        a_mu_hat_sec[:,i] = square(Reff_sec[:,i]) ./ sigma2_sec[:,i];
        b_mu_hat_sec[:,i] = Reff_sec[:,i] ./ sigma2_sec[:,i];
    }
    
    for (i in 1:j_third) {
        a_mu_hat_third[:,i] = square(Reff_third[:,i]) ./ sigma2_third[:,i];
        b_mu_hat_third[:,i] = Reff_third[:,i] ./ sigma2_third[:,i];
        a_mu_hat_omicron[:,i] = square(Reff_omicron[:,i]) ./ sigma2_omicron[:,i];
        b_mu_hat_omicron[:,i] = Reff_omicron[:,i] ./ sigma2_omicron[:,i];
    }
    
}

parameters {
    
    // macro and micro parameters 
    vector[K] bet;
    
    real<lower=0> theta_md;
    real<lower=0> theta_masks;
    
    // arrays of matrices that feature noisy macro estimates
    // array[j_first] matrix[N_first,K] mob_noisy_raw;
    // vector[total_N_p_sec * K] mob_sec_noisy_raw;
    // vector[total_N_p_third * K] mob_third_noisy_raw;
    
    matrix<lower=0,upper=1>[N_first,j_first] prop_md;
    vector<lower=0,upper=1>[total_N_p_sec] prop_md_sec;
    vector<lower=0,upper=1>[total_N_p_third] prop_md_third;
    
    matrix<lower=0,upper=1>[N_first,j_first] prop_masks;
    vector<lower=0,upper=1>[total_N_p_sec] prop_masks_sec;
    vector<lower=0,upper=1>[total_N_p_third] prop_masks_third;
    
    // import baseline R_I
    real<lower=0> R_I0;
    // captures the effective of quarantine on hotel workers
    real<lower=0,upper=1> import_ve_delta;
    // this is the import R effective used from opening (roughly 15/11/21)
    real<lower=0> R_I_omicron;
    
    // baseline and hierearchical RL parameters 
    real<lower=0> R_L;
    vector<lower=0>[j_total] R_Li;
    real<lower=0> sig;
    
    // import parameters 
    matrix<lower=0,upper=1>[N_first,j_first] brho;
    vector<lower=0,upper=1>[total_N_p_sec] brho_sec;
    vector<lower=0,upper=1>[total_N_p_third] brho_third;
    
    // voc parameters
    real<lower=0> additive_voc_effect_alpha;
    real<lower=0> additive_voc_effect_delta;
    real<lower=0> additive_voc_effect_omicron;
    
    // vaccine model parameters 
    vector<lower=0,upper=1>[total_N_p_third_blocks] ve_delta_tau;
    vector<lower=0,upper=1>[total_N_p_third_omicron_blocks] ve_omicron_tau;
    
    // real<lower=0,upper=1> sus_dep_factor;
    
    // parameters for the transition from Delta to Omicron 
    vector[j_third] tau_raw; 
    vector<lower=0>[j_third] r;
    vector<lower=0,upper=1>[j_third] m0; 
    vector<lower=0,upper=1>[j_third] m1; 
    
    // the simplex is equal to the number of periods of infections + 1
    simplex[7] phi_simplex; 
}

transformed parameters {
    
    // transform the ve from the fixed weekly inferred values to daily 
    vector<lower=0,upper=1>[total_N_p_third] ve_delta;
    vector<lower=0,upper=1>[total_N_p_third_omicron] ve_omicron;
    
    {
        int pos = 1;
        int pos_block = 1;
        int pos_c = 0; 
        
        int pos_omicron = 1; 
        int pos_block_omicron = 1; 
        int pos_omicron_c = 0; 
        
        for (i in 1:j_third){
            pos_c = 0;
            pos_omicron_c = 0;
            
            if (i == 1) {
                pos = 1;
                pos_block = 1;
                pos_omicron = 1;
                pos_block_omicron = 1;
            } else {
                pos = pos_starts_third[i-1] + 1;
                pos_block = pos_starts_third_blocks[i-1] + 1;
                pos_omicron = pos_starts_third_omicron[i-1] + 1;
                pos_block_omicron = pos_starts_third_omicron_blocks[i-1] + 1;
            }
        
            for (n in 1:N_third) {
                if (include_in_third[i][n] == 1) {
                    ve_delta[pos] = ve_delta_tau[pos_block];
                    pos_c += 1; 
                    if (pos_c == tau_vax_block_size) {
                        pos_c = 0; 
                        pos_block += 1;
                    }
                    pos += 1;
                    
                    if (include_in_omicron[i][n] == 1) {
                        ve_omicron[pos_omicron] = ve_omicron_tau[pos_block_omicron];
                        pos_omicron_c += 1;
                        if (pos_omicron_c == tau_vax_block_size) {
                            pos_omicron_c = 0; 
                            pos_block_omicron += 1;
                        }
                        pos_omicron += 1;
                    }
                }
            }
        }
    }
    
    // voc parameters (shifted by 1)
    real voc_effect_alpha = 1 + additive_voc_effect_alpha;
    real voc_effect_delta = 1 + additive_voc_effect_delta;
    real voc_effect_omicron = 1 + additive_voc_effect_omicron;
    
    // TP models 
    matrix[N_first,j_first] mu_hat;
    vector[total_N_p_sec] mu_hat_sec;
    vector[total_N_p_third] mu_hat_third;
    
    // micro distancing model
    matrix[N_first,j_first] md;
    vector[total_N_p_sec] md_sec;
    vector[total_N_p_third] md_third;
    
    // micro distancing model
    matrix[N_first,j_first] masks;
    vector[total_N_p_sec] masks_sec;
    vector[total_N_p_third] masks_third;
    
    // array[j_first] matrix[N_first,K] mob_noisy;
    // vector[total_N_p_sec * K] mob_sec_noisy;
    // vector[total_N_p_third * K] mob_third_noisy;    

    // first wave model
    for (i in 1:j_first) {
        real TP_local;
        real social_measures;
        real mob_input;
        
        for (n in 1:N_first) {
            if (include_in_first[i][n] == 1) {
                // mob_input = 0.0;
                // for (k in 1:K) {
                //     mob_noisy[i][n,k] = (
                //         mob_noisy_raw[i][n,k] * mob_std[i][n,k] + mob[i][n,k]
                //     );
                //     mob_input += mob_noisy[i][n,k] * bet[k];
                // }
                
                mob_input = mob[i][n,:] * (bet);
                md[n,i] = pow(1 + theta_md, -1 * prop_md[n,i]);
                masks[n,i] = pow(1 + theta_masks, -1 * prop_masks[n,i]);
                
                social_measures = (
                    (1 - policy[n]) + md[n,i] * policy[n] * masks[n,i]   
                ) * 2 * inv_logit(mob_input);
                
                TP_local = R_Li[map_to_state_index_first[i]] * social_measures;
                mu_hat[n,i] = brho[n,i] * R_I0 + (1 - brho[n,i]) * TP_local;
            }
        }
    }

    // second wave model
    for (i in 1:j_sec) {
        int pos;
        real TP_local;
        real social_measures;
        real mob_input; 
        
        if (i == 1) {
            pos = 1;
        } else {
            pos = pos_starts_sec[i-1] + 1;
        }
        
        for (n in 1:N_sec) {
            if (include_in_sec[i][n] == 1) {
                // mob_input = 0.0;
                // for (k in 1:K) {
                //     mob_sec_noisy[pos + k - 1] = (
                //         mob_sec_noisy_raw[pos + k - 1] * mob_sec_std[i][n,k] 
                //         + mob_sec[i][n,k] 
                //     );
                //     mob_input += mob_sec_noisy[pos + k - 1] * bet[k];
                // }
                
                mob_input = mob_sec[i][n,:] * (bet);
                
                md_sec[pos] = pow(1 + theta_md, -1 * prop_md_sec[pos]);
                masks_sec[pos] = pow(1 + theta_masks, -1 * prop_masks_sec[pos]);
                
                social_measures = (
                    (1 - policy_sec[n]) + md_sec[pos] * policy_sec[n] * masks_sec[pos]
                ) * 2 * inv_logit(mob_input);
                
                TP_local = R_Li[map_to_state_index_sec[i]] * social_measures; 
                mu_hat_sec[pos] = brho_sec[pos] * R_I0 + (1 - brho_sec[pos]) * TP_local;
                pos += 1;
            }
        }
    }

    // the import component of the Reff needs to be adjusted by the inferred risk of Delta 
    // as well as the effect of hotel quarantine and vaccinations (in delta phase) which 
    // we combine into a single reduction factor import_ve_delta. import_ve_omicron 
    // is the reduction following opening during the Delta-Omicron wave.
    real<lower=0> R_I = R_I0 * voc_effect_delta * import_ve_delta;
    
    // actual inflection point for the Omicron proportions 
    vector[j_third] tau;
    
    for (i in 1:j_third) {
        real mu_tau = 0.0;
        real sig_tau = 3.0;
        
        if (i == 8 || i == 6) {
            mu_tau = 70;  // based on a report for NT, there were only 2 Omicron cases by 15/12/21
        } else if (i == 3 || i == 5) {
            mu_tau = 60;  // based off report for SA, there were ~70% of cases by December 22/12/21
        } else {
            mu_tau = 30;  // for the other jurisdictions, let it be inferred earlier
        }
        tau[i] = mu_tau + sig_tau * tau_raw[i];
    }
    
    // Use a trick mentioned by Bob Carpenter on the Stan forums: 
    // (link https://groups.google.com/g/stan-users/c/04GSu-ql3vM)
    // to use a simplex X of dimension K+1 and simple 
    // transformation to a positive_ordered vector on (a, b), 
    // Y = (a + X[1] * (b - a), a + sum(X[1:2]) * (b - a), ..., a + sum(X[1:K] * (b - a))
    positive_ordered[6] phi = 0 + head(cumulative_sum(phi_simplex), 6) * (1 - 0);
    
    // third wave model
    for (i in 1:j_third) {
        // define these within the scope of the loop only
        int pos;
        int pos_omicron2;
        real social_measures; 
        real sus_dep_term;
        vector[6] sus_dep_comp;
        real prop_omicron; 
        int n_omicron; 
        real voc_ve_prod; 
        real TP_import;
        real mob_input;
        
        if (i == 1) {
            pos = 1;
            pos_omicron2 = 1;
        } else {
            pos = pos_starts_third[i-1] + 1;
            pos_omicron2 = pos_starts_third_omicron[i-1] + 1;
        }
        
        for (n in 1:N_third) {
            if (include_in_third[i][n] == 1) {
                // mob_input = 0.0;
                // for (k in 1:K) {
                //     mob_third_noisy[pos + k - 1] = (
                //         mob_third_noisy_raw[pos + k - 1] * mob_third_std[i][n,k] 
                //         + mob_third[i][n,k] 
                //     );
                //     mob_input += mob_third_noisy[pos + k - 1] * bet[k];
                // }
                
                mob_input = mob_third[i][n,:] * (bet);
                md_third[pos] = pow(1 + theta_md, -1 * prop_md_third[pos]);
                masks_third[pos] = pow(1 + theta_masks, -1 * prop_masks_third[pos]);
                
                social_measures = (
                    2 * inv_logit(mob_input) 
                    * md_third[pos]
                    * masks_third[pos]
                );  
                
                // calculate the effective proporiton infected
                sus_dep_comp[1] = phi[6] * prop_inf_14[n,i];
                sus_dep_comp[2] = phi[5] * prop_inf_28[n,i];
                sus_dep_comp[3] = phi[4] * prop_inf_42[n,i];
                sus_dep_comp[4] = phi[3] * prop_inf_72[n,i];
                sus_dep_comp[5] = phi[2] * prop_inf_102[n,i];
                sus_dep_comp[6] = phi[1] * prop_inf_132[n,i];
                    
                // total term is just the sum of the above
                sus_dep_term = 1.0 - sum(sus_dep_comp);

                if (n < omicron_start_day) {
                    voc_ve_prod = voc_effect_delta * ve_delta[pos];
                    TP_import = R_I;
                } else if (n < omicron_only_day) {
                    // number of days into omicron period 
                    n_omicron = n - omicron_start_day;
                    prop_omicron = sigmoid(
                        n_omicron, 
                        tau[map_to_state_index_third[i]], 
                        r[map_to_state_index_third[i]], 
                        m0[map_to_state_index_third[i]],
                        m1[map_to_state_index_third[i]]
                    );
                    
                    voc_ve_prod = (
                        prop_omicron
                        * voc_effect_omicron
                        * ve_omicron[pos_omicron2] 
                        + (1 - prop_omicron)
                        * voc_effect_delta
                        * ve_delta[pos]
                    );
                    
                    // adjust the import R by the proportion of cases and VoC's
                    TP_import = R_I_omicron * (
                        prop_omicron * voc_effect_omicron 
                        + (1 - prop_omicron) * voc_effect_delta
                    ); 
                    
                    pos_omicron2 += 1;  
                } else {
                    voc_ve_prod = voc_effect_omicron * ve_omicron[pos_omicron2];
                    // adjust the import R by the proportion of cases and VoC's
                    TP_import = R_I_omicron * voc_effect_omicron; 
                    
                    pos_omicron2 += 1;  
                }
                
                mu_hat_third[pos] = (
                    brho_third[pos]
                    * TP_import
                    + (1 - brho_third[pos]) 
                    * R_Li[map_to_state_index_third[i]]
                    * social_measures 
                    * voc_ve_prod
                ) * sus_dep_term;
                
                pos += 1;
            }
        }
    }
}

model {

    // indices for moving through parameter vectors
    int pos2_start;
    int pos2_end;
    int pos2;
    // this needs to be real for correct floating point division 
    real pos_omicron_c;   
    int pos_omicron2_start;
    int pos_omicron2_end;
    
    // define these within the scope of the loop only
    int pos;
    int pos_omicron2;
    real prop_omicron; 
    int n_omicron; 

    // variables for vax effects (reused for Delta and Omicron VEs)
    real mean_vax;
    real var_vax = 0.00005;     
    real a_vax_scalar;
    real b_vax_scalar;
    
    // index arrays for vectorising the model which makes it more efficient
    array[N_sec] int idxs_sec;
    array[N_third] int idxs_third;
    int pos_idxs;

    // priors
    // mobility, micro
    bet ~ normal(0, 1);
    // for (i in 1:j_first) {
    //     for (k in 1:5) {
    //         mob_noisy_raw[i][:,k] ~ normal(0, 1);
    //     }
    // }
    // mob_sec_noisy_raw ~ normal(0, 1);
    // mob_third_noisy_raw ~ normal(0, 1);
    
    theta_md ~ lognormal(0, 0.5);
    theta_masks ~ lognormal(0, 0.5);
    
    // third wave transition parameters 
    // r_mean ~ gamma(20, 40);     // mean of 0.75 
    // r_sig ~ exponential(500);   // mean of 1/200 
    real r_sig = 0.025;
    real r_mean = 0.8;
    r ~ gamma(square(r_mean) / r_sig, r_mean / r_sig);
    
    for (i in 1:j_third){
        if (i == 8 || i == 6 || i == 3) {
            m0[i] ~ beta(5, 5);
        } else {
            m0[i] ~ beta(5, 95);
        }
    }
    
    tau_raw ~ normal(0, 1);
    
    // assume final proportion of ~97%
    m1 ~ beta(5 * 97, 5 * 3);
    
    // gives full priors of 1 + Gamma() for each VoC effect
    additive_voc_effect_alpha ~ gamma(square(0.4) / 0.05, 0.4 / 0.05);
    additive_voc_effect_delta ~ gamma(square(2.0) / 0.05, 2.0 / 0.05);
    additive_voc_effect_omicron ~ gamma(square(2.0) / 0.05, 2.0 / 0.05);

    // Even though this prior is on the transformed parameter phi which depends on 
    // phi_simplex, we don't need a Jacobian adjustment as the Jacobian will have unit 
    // determinant due to the fixed (a, b) = (0, 1). 
    // phi[4:6] ~ beta(9, 2);
    // phi[1:3] ~ beta(2, 2);
    phi[6] ~ beta(9, 2);
    phi[2:5] ~ beta(2, 2);
    phi[1] ~ beta(2, 9);
    
    // this effect is the informed scaling (adjustment in the import ve) by the previous 
    // estimates used inside the generative model. It is the product of hotel worker 
    // vaccination efficacy and coverage of vaccinated hotel workers. Informed by TKI mid-2021
    import_ve_delta ~ beta(20.5, 105);
    
    R_I0 ~ gamma(square(0.3) / 0.2, 0.3 / 0.2);
    R_I_omicron ~ gamma(square(0.3) / 0.2, 0.3 / 0.2);

    // hierarchical model for the baseline RL
    R_L ~ gamma(square(1.7) / 0.05, 1.7 / 0.05);
    sig ~ exponential(100);
    R_Li ~ gamma(square(R_L) / sig, R_L / sig);

    // first wave model
    for (i in 1:j_first) { 
        prop_md[:,i] ~ beta(1 + count_md[i][:], 1 + respond_md[i][:] - count_md[i][:]);
        
        prop_masks[:,i] ~ beta(
            1 + count_masks[i][:], 
            1 + respond_masks[i][:] - count_masks[i][:]
        );
        
        brho[:,i] ~ beta(0.5 + imported[:,i], 0.5 + local[:,i]);
        
        // likelihood
        mu_hat[:,i] ~ gamma(a_mu_hat[:,i], b_mu_hat[:,i]);
    }

    // second wave model
    for (i in 1:j_sec){
        pos_idxs = 1;
        
        if (i == 1){
            pos2_start = 1;
            pos2_end = pos_starts_sec[i];
        } else {
            pos2_start = pos_starts_sec[i-1] + 1;
            pos2_end = pos_starts_sec[i];
        }
        
        // create an array for indexing the proportion terms
        for (n in 1:N_sec){ 
            if (include_in_sec[i][n] == 1) {
                idxs_sec[pos_idxs] = n;
                pos_idxs += 1;  
            }
        }
        
        prop_md_sec[pos2_start:pos2_end] ~ beta(
            1 + count_md_sec[i][idxs_sec[1:pos_idxs-1]], 
            1 + respond_md_sec[i][idxs_sec[1:pos_idxs-1]]
                - count_md_sec[i][idxs_sec[1:pos_idxs-1]]
        );
        
        prop_masks_sec[pos2_start:pos2_end] ~ beta(
            1 + count_masks_sec[i][idxs_sec[1:pos_idxs-1]], 
            1 + respond_masks_sec[i][idxs_sec[1:pos_idxs-1]]
                - count_masks_sec[i][idxs_sec[1:pos_idxs-1]]
        );
        
        brho_sec[pos2_start:pos2_end] ~ beta(
            0.5 + imported_sec[idxs_sec[1:pos_idxs-1],i], 
            0.5 + local_sec[idxs_sec[1:pos_idxs-1],i]
        );

        // likelihood
        mu_hat_sec[pos2_start:pos2_end] ~ gamma(
            a_mu_hat_sec[idxs_sec[1:pos_idxs-1],i], 
            b_mu_hat_sec[idxs_sec[1:pos_idxs-1],i]
        );
    }
    
    // VE model 
    int pos_block = 1;
    int pos_c = 0; 
    int pos_block_omicron = 1; 
    int heterogeneity_in_vax_count = 1; 
    int heterogeneity_in_vax = 1; 
    
    for (i in 1:j_third){
        pos_c = 0;
        pos_omicron_c = 0;
        
        if (i == 1){
            pos_block = 1;
            pos_block_omicron = 1;
        } else {
            pos_block = pos_starts_third_blocks[i-1] + 1;
            pos_block_omicron = pos_starts_third_omicron_blocks[i-1] + 1;
        }
        
        // reset heterogeneity terms 
        heterogeneity_in_vax = 1;
        heterogeneity_in_vax_count += 1;
        
        //reverse the array
        for (n in 1:N_third){
            if (include_in_third[i][n] == 1){
                if (pos_c == 0){    
                    if (n < tau_vax_block_size) {
                        mean_vax = ve_delta_data[i][n];
                    } else if (include_in_third[i][n-1] == 0){
                        mean_vax = ve_delta_data[i][n];
                    } else {
                        mean_vax = mean(ve_delta_data[i][n-tau_vax_block_size+1:n]);
                    }
                    
                    // if (heterogeneity_in_vax == 1) {
                    //     a_vax_scalar = 100; 
                    //     b_vax_scalar = 2;
                    // } else 
                    if (mean_vax * (1 - mean_vax) > var_vax) {
                        a_vax_scalar = mean_vax * (
                            mean_vax * (1 - mean_vax) / var_vax - 1
                        );
                        b_vax_scalar = (1 - mean_vax) * (
                            mean_vax * (1 - mean_vax) / var_vax - 1
                        );
                    } else {
                        // if we are close to 1 or 0, tight prior value        
                        if (mean_vax > 0.98) {
                            a_vax_scalar = 100;
                            b_vax_scalar = 2;
                        } else if (mean_vax < 0.02) {
                            a_vax_scalar = 2;
                            b_vax_scalar = 100;
                        }
                    }
                    ve_delta_tau[pos_block] ~ beta(a_vax_scalar, b_vax_scalar);
                }
                    
                pos_c += 1; 
                
                if (pos_c == tau_vax_block_size) {
                    pos_c = 0; 
                    pos_block += 1;
                    heterogeneity_in_vax_count += 1;
                    if (heterogeneity_in_vax_count > 5) {
                        heterogeneity_in_vax = 0;
                    }
                }
            }
            
            // reset heterogeneity terms 
            heterogeneity_in_vax = 1;
            heterogeneity_in_vax_count += 1;
            
            if (include_in_omicron[i][n] == 1){
                if (pos_omicron_c == 0){    
                    if (n < tau_vax_block_size) {
                        mean_vax = ve_omicron_data[i][n];
                    } else if (include_in_omicron[i][n-1] == 0) {
                        mean_vax = ve_omicron_data[i][n];
                    }else {
                        mean_vax = mean(ve_omicron_data[i][n-tau_vax_block_size+1:n]);
                    }
                    
                    if (mean_vax*(1-mean_vax) > var_vax) {
                        a_vax_scalar = mean_vax * (
                            mean_vax * (1 - mean_vax) / var_vax - 1
                        );
                        b_vax_scalar = (1 - mean_vax) * (
                            mean_vax * (1 - mean_vax) / var_vax - 1);
                    } else {
                        // if we are close to 1 or 0, tight prior on that       
                        if (mean_vax > 0.98) {
                            a_vax_scalar = 100;
                            b_vax_scalar = 2;
                        } else if (mean_vax < 0.02) {
                            a_vax_scalar = 2;
                            b_vax_scalar = 100;
                        }
                    }
                    ve_omicron_tau[pos_block_omicron] ~ beta(a_vax_scalar, b_vax_scalar);
                }
                    
                pos_omicron_c += 1; 
                if (pos_omicron_c == tau_vax_block_size) {
                    pos_omicron_c = 0; 
                    pos_block_omicron += 1;
                }
            }
        }
    }
    
    // third wave model
    for (i in 1:j_third){
        pos_idxs = 1;
        
        if (i == 1){
            pos2_start = 1;
            pos2_end = pos_starts_third[i];
        } else {
            pos2_start = pos_starts_third[i-1] + 1;
            pos2_end = pos_starts_third[i];
        }
        
        // create an array for indexing parameters, this will contain the 
        // days, n, that the wave is happening (i.e. idxs_third[1] is the first
        // day for the jurisdictions 3rd wave fitting).
        for (n in 1:N_third){ 
            if (include_in_third[i][n] == 1) {
                idxs_third[pos_idxs] = n;
                pos_idxs += 1;  
            }
        }
        
        prop_md_third[pos2_start:pos2_end] ~ beta(
            0.5 + count_md_third[i][idxs_third[1:pos_idxs-1]], 
            0.5 + respond_md_third[i][idxs_third[1:pos_idxs-1]]
                - count_md_third[i][idxs_third[1:pos_idxs-1]]
        );
        
        prop_masks_third[pos2_start:pos2_end] ~ beta(
            1 + count_masks_third[i][idxs_third[1:pos_idxs-1]], 
            1 + respond_masks_third[i][idxs_third[1:pos_idxs-1]]
                - count_masks_third[i][idxs_third[1:pos_idxs-1]]
        );
        
        brho_third[pos2_start:pos2_end] ~ beta(
            0.5 + imported_third[idxs_third[1:pos_idxs-1],i], 
            0.5 + local_third[idxs_third[1:pos_idxs-1],i]
        );
    }
    
    int Reff_switchover_day;
    
    for (i in 1:j_third){
        if (i == 1){
            pos = 1;
        } else {
            pos = pos_starts_third[i-1] + 1;
        }
        
        if (i == 8) {
            Reff_switchover_day = omicron_start_day + 75;
        } else {
            Reff_switchover_day = omicron_start_day + 15;
        }
        
        for (n in 1:N_third){
            if (include_in_third[i][n] == 1){
                if (n < Reff_switchover_day) {
                    mu_hat_third[pos] ~ gamma(
                        a_mu_hat_third[n,i], 
                        b_mu_hat_third[n,i]
                    );
                    
                    pos += 1;
                } else {
                    mu_hat_third[pos] ~ gamma(
                        a_mu_hat_omicron[n,i], 
                        b_mu_hat_omicron[n,i]
                    );
                    
                    pos += 1;
                }
            }
        }
    }
}

generated quantities {
    
    // generate a TP independently for delta and omicron waves for comparisons against the 
    // independent estimates
    vector[total_N_p_third] mu_hat_delta_only;
    vector[total_N_p_third_omicron] mu_hat_omicron_only;
    
    // generate posterior predictive estimates of the factors that adjust the TP 
    vector[total_N_p_third] micro_factor;
    vector[total_N_p_third] macro_factor;
    vector[total_N_p_third] sus_dep_factor;
    
    vector[total_N_p_third] macro_level_data;
    vector[total_N_p_third] macro_level_inferred;
    
    for (i in 1:j_third) {
        // define these within the scope of the loop only
        int pos;
        int pos_omicron2;
        real social_measures; 
        real sus_dep;
        vector[6] sus_dep_comp;
        real voc_ve_prod;
        real mob_input; 
        real mob_input_data; 
        
        if (i == 1) {
            pos = 1;
            pos_omicron2 = 1;
        } else {
            pos = pos_starts_third[i-1] + 1;
            pos_omicron2 = pos_starts_third_omicron[i-1] + 1;
        }
        
        for (n in 1:N_third) {
            if (include_in_third[i][n] == 1) {
                // calculate the effective proporiton infected
                sus_dep_comp[1] = phi[6] * prop_inf_14[n,i];
                sus_dep_comp[2] = phi[5] * prop_inf_28[n,i];
                sus_dep_comp[3] = phi[4] * prop_inf_42[n,i];
                sus_dep_comp[4] = phi[3] * prop_inf_72[n,i];
                sus_dep_comp[5] = phi[2] * prop_inf_102[n,i];
                sus_dep_comp[6] = phi[1] * prop_inf_132[n,i];
                    
                // total term is just the sum of the above
                sus_dep = 1.0 - sum(sus_dep_comp);
                sus_dep_factor[pos] = sus_dep;
                
                micro_factor[pos] = md_third[pos];
                macro_factor[pos] = 2 * inv_logit(mob_third[i][n,:] * (bet));
                
                // mob_input = 0.0;
                // for (k in 1:K) {
                //     mob_input += mob_third_noisy[pos + k - 1] * bet[k];
                // }
                
                mob_input_data = mob_third[i][n,:] * bet;
                
                macro_level_data[pos] = 2 * inv_logit(mob_input_data);
                // macro_level_inferred[pos] = 2 * inv_logit(mob_input);
                
                social_measures = (
                    // macro_level_inferred[pos]
                    macro_level_data[pos]
                    * md_third[pos]
                    * masks_third[pos]
                );  

                if (n < omicron_start_day) {
                    voc_ve_prod = voc_effect_delta * ve_delta[pos];
                    
                    mu_hat_delta_only[pos] = (
                        brho_third[pos] 
                        * R_I
                        + (1 - brho_third[pos])
                        * R_Li[map_to_state_index_third[i]]
                        * social_measures 
                        * voc_ve_prod
                    ) * sus_dep;
                    
                    pos += 1;
                } else {
                    voc_ve_prod = voc_effect_delta * ve_delta[pos];
                    
                    mu_hat_delta_only[pos] = (
                        brho_third[pos] 
                        * R_I_omicron
                        * voc_effect_delta 
                        + (1 - brho_third[pos])
                        * R_Li[map_to_state_index_third[i]]
                        * social_measures 
                        * voc_ve_prod
                    ) * sus_dep;
                    
                    voc_ve_prod = voc_effect_omicron * ve_omicron[pos_omicron2];
                    
                    mu_hat_omicron_only[pos_omicron2] = (
                        brho_third[pos] 
                        * R_I_omicron
                        * voc_effect_omicron
                        + (1 - brho_third[pos])
                        * R_Li[map_to_state_index_third[i]]
                        * social_measures 
                        * voc_ve_prod
                    ) * sus_dep;
                    
                    pos += 1;
                    pos_omicron2 += 1;  
                }
            }
        }
    }
}