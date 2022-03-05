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
    int N;
    int K;
    int j_first_wave;
    matrix[N,j_first_wave] Reff;
    matrix[N,K] Mob[j_first_wave];
    matrix[N,K] Mob_std[j_first_wave];
    matrix[N,j_first_wave] sigma2;
    vector[N] policy;
    matrix[N,j_first_wave] local;
    matrix[N,j_first_wave] imported;
    
    // second wave data 
    int N_sec_wave;
    int j_sec_wave;
    matrix[N_sec_wave,j_sec_wave] Reff_sec_wave;
    matrix[N_sec_wave,K] Mob_sec_wave[j_sec_wave];
    matrix[N_sec_wave,K] Mob_sec_wave_std[j_sec_wave];
    matrix[N_sec_wave,j_sec_wave] sigma2_sec_wave;
    vector[N_sec_wave] policy_sec_wave;
    matrix[N_sec_wave,j_sec_wave] local_sec_wave;
    matrix[N_sec_wave,j_sec_wave] imported_sec_wave;
    int apply_alpha_sec_wave[N_sec_wave];
    
    // third wave data 
    int N_third_wave;
    int j_third_wave;
    matrix[N_third_wave,j_third_wave] Reff_third_wave;
    matrix[N_third_wave,K] Mob_third_wave[j_third_wave];
    matrix[N_third_wave,K] Mob_third_wave_std[j_third_wave];
    matrix[N_third_wave,j_third_wave] sigma2_third_wave;
    vector[N_third_wave] policy_third_wave;
    matrix[N_third_wave,j_third_wave] local_third_wave;
    matrix[N_third_wave,j_third_wave] imported_third_wave;
    
    // micro data
    vector[N] count_md[j_first_wave];
    vector[N] respond_md[j_first_wave];
    vector[N_sec_wave] count_md_sec_wave[j_sec_wave];
    vector[N_sec_wave] respond_md_sec_wave[j_sec_wave];
    vector[N_third_wave] count_md_third_wave[j_third_wave];
    vector[N_third_wave] respond_md_third_wave[j_third_wave];
    
    // masks data 
    vector[N] count_masks[j_first_wave];
    vector[N] respond_masks[j_first_wave];
    vector[N_sec_wave] count_masks_sec_wave[j_sec_wave];
    vector[N_sec_wave] respond_masks_sec_wave[j_sec_wave];
    vector[N_third_wave] count_masks_third_wave[j_third_wave];
    vector[N_third_wave] respond_masks_third_wave[j_third_wave];
    
    // vectors that map to the correct indices based on j_total 
    int map_to_state_index_first[j_first_wave];
    int map_to_state_index_sec[j_sec_wave];
    int map_to_state_index_third[j_third_wave];
    
    // ints for moving through the total parameter vectors in the second or 
    // third waves
    int total_N_p_sec;
    int total_N_p_third;
    
    // bool arrays for when to include data 
    vector[N] include_in_first_wave[j_first_wave];
    vector[N_sec_wave] include_in_sec_wave[j_sec_wave];
    vector[N_third_wave] include_in_third_wave[j_third_wave];
    
    // this is used to index starting points in include_in_XX_wave 
    int pos_starts_sec[j_sec_wave];
    int pos_starts_third[j_third_wave];
    // vax data for each strain
    vector[N_third_wave] ve_delta_data[j_third_wave];
    // omicron ve time series is the same length as delta but we fit to the 
    // appropriate range
    vector[N_third_wave] ve_omicron_data[j_third_wave];     
    // data for handling omicron 
    int omicron_start_day;
    int omicron_dominance_day;
    vector[N_third_wave] include_in_omicron_wave[j_third_wave];
    int total_N_p_third_omicron;
    int pos_starts_third_omicron[j_third_wave];
    
    int tau_vax_block_size; 
    int total_N_p_third_blocks;
    int pos_starts_third_blocks[j_third_wave];
    int total_N_p_third_omicron_blocks;
    int pos_starts_third_omicron_blocks[j_third_wave];
    
    int pop_size_array[j_total];
    int heterogeneity_start_day;
    
    real p_detect_delta; 
    real p_detect_omicron; 
    
}

transformed data {
    
    // for now we just calculate the cumulative number of cases in the
    // third wave as pre third wave cases were negligible
    vector[N_third_wave] local_third_wave_tmp;
    vector[N_third_wave] cumulative_local_third;
    // to be used as the accurate number of cases when determining the import fraction
    matrix[N_third_wave,j_third_wave] local_third_wave_ca_adjusted;
    matrix[N_third_wave,j_third_wave] proportion_infected;
    
    // shape and scale for the likelihood in each wave
    matrix[N,j_first_wave] a_mu_hat;
    matrix[N,j_first_wave] b_mu_hat;
    matrix[N_sec_wave,j_sec_wave] a_mu_hat_sec_wave;
    matrix[N_sec_wave,j_sec_wave] b_mu_hat_sec_wave;
    matrix[N_third_wave,j_third_wave] a_mu_hat_third_wave;
    matrix[N_third_wave,j_third_wave] b_mu_hat_third_wave;
    
    for (i in 1:j_third_wave){
        local_third_wave_tmp = local_third_wave[:,i];
        for (n in 1:N_third_wave) { 
            // scale up local case count by assumed ascertainment -5 (mean incubation 
            // period) days as case ascertainment is assumed to scale up on the 15/12/2021
            // but that relates only to detections of actual cases
            if (n <= omicron_dominance_day - 5) {
                local_third_wave_tmp[n] *= 1 / p_detect_delta;
            } else {
                local_third_wave_tmp[n] *= 1 / p_detect_omicron;
            }
        }
        
        local_third_wave_ca_adjusted[:,i] = local_third_wave_tmp; 
        cumulative_local_third = cumulative_sum(local_third_wave_tmp);
        
        for (n in 1:N_third_wave) {
            // can't have more cases than the observed population
            proportion_infected[n,i] = fmin(
               (1.0 * cumulative_local_third[n])
                    / (1.0 * pop_size_array[map_to_state_index_third[i]]), 
                1.0
            );
        }
    }

    // compute the shape and scale for the likelihood 
    for (i in 1:j_first_wave) {
        a_mu_hat[:,i] = square(Reff[:,i]) ./ sigma2[:,i];
        b_mu_hat[:,i] = Reff[:,i] ./ sigma2[:,i];
    }
    
    for (i in 1:j_sec_wave) {
        a_mu_hat_sec_wave[:,i] = square(Reff_sec_wave[:,i]) ./ sigma2_sec_wave[:,i];
        b_mu_hat_sec_wave[:,i] = Reff_sec_wave[:,i] ./ sigma2_sec_wave[:,i];
    }
    
    for (i in 1:j_third_wave) {
        a_mu_hat_third_wave[:,i] = square(Reff_third_wave[:,i]) ./ sigma2_third_wave[:,i];
        b_mu_hat_third_wave[:,i] = Reff_third_wave[:,i] ./ sigma2_third_wave[:,i];
    }
}

parameters {
    
    // macro and micro parameters 
    vector[K] bet;
    
    real<lower=0> theta_md;
    // real<lower=0> theta_masks;
    
    matrix<lower=0,upper=1>[N,j_first_wave] prop_md;
    vector<lower=0,upper=1>[total_N_p_sec] prop_md_sec_wave;
    vector<lower=0,upper=1>[total_N_p_third] prop_md_third_wave;
    
    // matrix<lower=0,upper=1>[N,j_first_wave] prop_masks;
    // vector<lower=0,upper=1>[total_N_p_sec] prop_masks_sec_wave;
    // vector<lower=0,upper=1>[total_N_p_third] prop_masks_third_wave;
    
    // baseline and hierearchical RL parameters 
    real<lower=0> R_I;
    real<lower=0> R_I_omicron;
    real<lower=0> R_L;
    vector<lower=0>[j_total] R_Li;
    real<lower=0> sig;
    
    // import parameters 
    matrix<lower=0,upper=1>[N,j_first_wave] brho;
    vector<lower=0,upper=1>[total_N_p_sec] brho_sec_wave;
    vector<lower=0,upper=1>[total_N_p_third] brho_third_wave;
    
    // voc parameters
    real<lower=0> additive_voc_effect_alpha;
    real<lower=0> additive_voc_effect_delta;
    real<lower=0> additive_voc_effect_omicron;
    
    // vaccine model parameters 
    vector<lower=0,upper=1>[total_N_p_third_blocks] ve_delta_tau;
    vector<lower=0,upper=1>[total_N_p_third_omicron_blocks] ve_omicron_tau;
    
    real<lower=0,upper=1> susceptible_depletion_factor;
    
    // parameters for the transition from Delta to Omicron 
    vector[j_third_wave] tau_raw; 
    vector<lower=0>[j_third_wave] r;
    vector<lower=0,upper=1>[j_third_wave] m0; 
    vector<lower=0,upper=1>[j_third_wave] m1; 
    
}

transformed parameters {
    
    // transform the ve
    vector<lower=0,upper=1>[total_N_p_third] ve_delta;
    vector<lower=0,upper=1>[total_N_p_third_omicron] ve_omicron;
    
    //reverse the ordering of the weekly vax effects in a local scope
    {
        int pos = 1;
        int pos_block = 1;
        int pos_counter = 0; 
        
        int pos_omicron = 1; 
        int pos_block_omicron = 1; 
        int pos_omicron_counter = 0; 
        
        for (i in 1:j_third_wave){
            pos_counter = 0;
            pos_omicron_counter = 0;
            
            if (i == 1){
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
            
            //reverse the array
            for (n in 1:N_third_wave){
                if (include_in_third_wave[i][n] == 1){
                    ve_delta[pos] = ve_delta_tau[pos_block];
                    pos_counter += 1; 
                    if (pos_counter == tau_vax_block_size) {
                        pos_counter = 0; 
                        pos_block += 1;
                    }
                    pos += 1;
                    
                    if (include_in_omicron_wave[i][n] == 1) {
                        ve_omicron[pos_omicron] = ve_omicron_tau[pos_block_omicron];
                        pos_omicron_counter += 1;
                        if (pos_omicron_counter == tau_vax_block_size) {
                            pos_omicron_counter = 0; 
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
    matrix[N,j_first_wave] mu_hat;
    vector[total_N_p_sec] mu_hat_sec_wave;
    vector[total_N_p_third] mu_hat_third_wave;
    
    // micro distancing model
    matrix[N,j_first_wave] md;
    vector[total_N_p_sec] md_sec_wave;
    vector[total_N_p_third] md_third_wave;
    
    // micro distancing model
    // matrix[N,j_first_wave] masks;
    // vector[total_N_p_sec] masks_sec_wave;
    // vector[total_N_p_third] masks_third_wave;
    
    vector[j_third_wave] tau = 25 + 2 * tau_raw; 

    // first wave model
    for (i in 1:j_first_wave) {
        real TP_local;
        real social_measures;
        
        for (n in 1:N){
            if (include_in_first_wave[i][n] == 1){
                md[n,i] = pow(1 + theta_md, -1 * prop_md[n,i]);
                // masks[n,i] = pow(1 + theta_masks, -1 * prop_masks[n,i]);
                // social_measures = (
                //     (1 - policy[n]) + md[n,i] * policy[n]   
                // ) * masks[n,i] * 2 * inv_logit(Mob[i][n,:] * (bet));
                social_measures = (
                    (1 - policy[n]) + md[n,i] * policy[n]   
                ) * 2 * inv_logit(Mob[i][n,:] * (bet));
                
                TP_local = R_Li[map_to_state_index_first[i]] * social_measures;
                mu_hat[n,i] = brho[n,i] * R_I + (1 - brho[n,i]) * TP_local;
            }
        }
    }

    // second wave model
    for (i in 1:j_sec_wave){
        int pos;
        real TP_local;
        real social_measures;
        
        if (i == 1){
            pos = 1;
        } else {
            pos = pos_starts_sec[i-1] + 1;
        }
        
        for (n in 1:N_sec_wave){
            if (include_in_sec_wave[i][n] == 1){
                md_sec_wave[pos] = pow(1 + theta_md, -1 * prop_md_sec_wave[pos]);
                // masks_sec_wave[pos] = pow(1 + theta_masks, -1 * prop_masks_sec_wave[pos]);
                // social_measures = (
                //     (1 - policy_sec_wave[n]) + md_sec_wave[pos] * policy_sec_wave[n]
                // ) * 2 * masks_sec_wave[pos] * inv_logit(Mob_sec_wave[i][n,:] * (bet));
                social_measures = (
                    (1 - policy_sec_wave[n]) + md_sec_wave[pos] * policy_sec_wave[n]
                ) * 2 * inv_logit(Mob_sec_wave[i][n,:] * (bet));
                
                TP_local = R_Li[map_to_state_index_sec[i]] * social_measures; 
                mu_hat_sec_wave[pos] = brho_sec_wave[pos] * R_I 
                    + (1 - brho_sec_wave[pos]) * TP_local;
                pos += 1;
            }
        }
    }

    // third wave model
    for (i in 1:j_third_wave){
        // define these within the scope of the loop only
        int pos;
        int pos_omicron2;
        real TP_local;
        real social_measures; 
        real voc_vacc_product; 
        real susceptible_depletion_term;
        real prop_omicron_to_delta; 
        int n_omicron; 
        real R_I_tmp; 
        
        if (i == 1){
            pos = 1;
            pos_omicron2 = 1;
        } else {
            pos = pos_starts_third[i-1] + 1;
            pos_omicron2 = pos_starts_third_omicron[i-1] + 1;
        }
        
        for (n in 1:N_third_wave){
            if (include_in_third_wave[i][n] == 1){
                md_third_wave[pos] = pow(
                    1 + theta_md, -1 * prop_md_third_wave[pos]
                );
                // masks_third_wave[pos] = pow(
                //     1 + theta_masks, -1 * prop_masks_third_wave[pos]
                // );

                if (n <= omicron_start_day){
                    voc_vacc_product = voc_effect_delta * ve_delta[pos];
                    R_I_tmp = R_I; 
                } else {
                    R_I_tmp = R_I_omicron; 
                    // number of days into omicron period 
                    n_omicron = n - omicron_start_day;
                    // proportion of omicron
                    if (
                        map_to_state_index_third[i] == 3 || 
                        map_to_state_index_third[i] == 6 || 
                        map_to_state_index_third[i] == 8
                    ) {
                        prop_omicron_to_delta = m1[map_to_state_index_third[i]];
                    } else {
                        prop_omicron_to_delta = sigmoid(
                            n_omicron, 
                            tau[map_to_state_index_third[i]], 
                            r[map_to_state_index_third[i]], 
                            m0[map_to_state_index_third[i]], 
                            m1[map_to_state_index_third[i]]
                        );
                    }
                    
                    voc_vacc_product = prop_omicron_to_delta
                        * voc_effect_omicron
                        * ve_omicron[pos_omicron2]
                        + (1 - prop_omicron_to_delta)
                        * voc_effect_delta
                        * ve_delta[pos];
                        
                    pos_omicron2 += 1;
                }
                
                // social_measures = 2 * inv_logit(Mob_third_wave[i][n,:] * (bet)) 
                //     * md_third_wave[pos] 
                //     * masks_third_wave[pos];
                social_measures = 2 * inv_logit(Mob_third_wave[i][n,:] * (bet)) 
                    * md_third_wave[pos];
                    
                susceptible_depletion_term = 
                    1 - susceptible_depletion_factor * proportion_infected[n,i];
                    
                TP_local = R_Li[map_to_state_index_third[i]]
                    * voc_vacc_product
                    * social_measures;
                    
                mu_hat_third_wave[pos] = (
                    brho_third_wave[pos] * R_I_tmp 
                    + (1 - brho_third_wave[pos]) * TP_local
                ) * susceptible_depletion_term;
                
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
    real pos_omicron_counter;   
    int pos_omicron2_start;
    int pos_omicron2_end;

    // variables for vax effects (reused for Delta and Omicron VEs)
    real mean_vax;
    real var_vax_delta = 0.00005;     
    real var_vax_omicron = 0.00005;
    // real var_vax_delta = 0.0005;     
    // real var_vax_omicron = 0.0005;
    real a_vax_scalar;
    real b_vax_scalar;
    
    // index arrays for vectorising the model which makes it more efficient
    array[N_sec_wave] int idxs_sec;
    array[N_third_wave] int idxs_third;
    int pos_idxs;

    // priors
    // mobility, micro
    bet ~ std_normal();
    theta_md ~ exponential(5);
    // theta_masks ~ exponential(5);
    
    // third wave transition parameters 
    // m0 ~ uniform(0, 0.1);     // initial Omicron proportion is low 
    // m1 ~ uniform(0.90, 1);     // long term Omicron proportion 
    // r ~ gamma(3*5, 3*20);       // median of 0.2
    r ~ gamma(20, 40);       // median of 0.2
    
    tau_raw ~ std_normal();
    m0 ~ beta(5, 95);
    m1 ~ beta(3 * 95, 3 * 5);
    
    // gives full priors of 1 + Gamma() for each VoC effect
    additive_voc_effect_alpha ~ gamma(square(0.4) / 0.05, 0.4 / 0.05);
    additive_voc_effect_delta ~ gamma(square(2.0) / 0.05, 2.0 / 0.05);
    additive_voc_effect_omicron ~ gamma(
        square(additive_voc_effect_delta) / 0.05, additive_voc_effect_delta / 0.05
    );

    susceptible_depletion_factor ~ beta(2, 2);

    // hierarchical model for the baseline RL
    R_L ~ gamma(square(1.7) / 0.005, 1.7 / 0.005);
    R_I ~ gamma(square(0.5) / 0.2, 0.5 / 0.2);
    R_I_omicron ~ gamma(square(0.5) / 0.2, 0.5 / 0.2);
    sig ~ exponential(250);
    R_Li ~ gamma(square(R_L) / sig, R_L / sig);

    // first wave model
    for (i in 1:j_first_wave) { 
        prop_md[:,i] ~ beta(
            1 + count_md[i][:], 1 + respond_md[i][:] - count_md[i][:]
        );
        
        // prop_masks[:,i] ~ beta(
        //     1 + count_masks[i][:], 1 + respond_masks[i][:] - count_masks[i][:]
        // );
        
        brho[:,i] ~ beta(
            1 + imported[:,i], 1 + local[:,i]
        );
        
        // likelihood
        mu_hat[:,i] ~ gamma(a_mu_hat[:,i], b_mu_hat[:,i]);
    }

    // second wave model
    for (i in 1:j_sec_wave){
        pos_idxs = 1;
        
        if (i == 1){
            pos2_start = 1;
            pos2_end = pos_starts_sec[i];
        } else {
            pos2_start = pos_starts_sec[i-1] + 1;
            pos2_end = pos_starts_sec[i];
        }
        
        // create an array for indexing the proportion terms
        for (n in 1:N_sec_wave){ 
            if (include_in_sec_wave[i][n] == 1){
                idxs_sec[pos_idxs] = n;
                pos_idxs += 1;  
            }
        }
        
        prop_md_sec_wave[pos2_start:pos2_end] ~ beta(
            1 + count_md_sec_wave[i][idxs_sec[1:pos_idxs-1]], 
            1 + respond_md_sec_wave[i][idxs_sec[1:pos_idxs-1]]
                - count_md_sec_wave[i][idxs_sec[1:pos_idxs-1]]
        );
        
        // prop_masks_sec_wave[pos2_start:pos2_end] ~ beta(
        //     1 + count_masks_sec_wave[i][idxs_sec[1:pos_idxs-1]], 
        //     1 + respond_masks_sec_wave[i][idxs_sec[1:pos_idxs-1]]
        //         - count_masks_sec_wave[i][idxs_sec[1:pos_idxs-1]]
        // );
        
        brho_sec_wave[pos2_start:pos2_end] ~ beta(
            1 + imported_sec_wave[idxs_sec[1:pos_idxs-1],i], 
            1 + local_sec_wave[idxs_sec[1:pos_idxs-1],i]
        );

        // likelihood
        mu_hat_sec_wave[pos2_start:pos2_end] ~ gamma(
            a_mu_hat_sec_wave[idxs_sec[1:pos_idxs-1],i], 
            b_mu_hat_sec_wave[idxs_sec[1:pos_idxs-1],i]
        );
    }
    
    
    // VE model 
    int pos_block = 1;
    int pos_counter = 0; 
    int pos_block_omicron = 1; 
    mean_vax = 0.0;
    
    for (i in 1:j_third_wave){
        pos_counter = 0;
        pos_omicron_counter = 0;
        
        if (i == 1){
            pos_block = 1;
            pos_block_omicron = 1;
        } else {
            pos_block = pos_starts_third_blocks[i-1] + 1;
            pos_block_omicron = pos_starts_third_omicron_blocks[i-1] + 1;
        }
        
        //reverse the array
        for (n in 1:N_third_wave){
            if (include_in_third_wave[i][n] == 1){
                if (pos_counter == 0){    
                    if (n < tau_vax_block_size) {
                        mean_vax = ve_delta_data[i][n];
                    } else if (include_in_third_wave[i][n-1] == 0){
                        mean_vax = ve_delta_data[i][n];
                    } else {
                        mean_vax = mean(ve_delta_data[i][n-tau_vax_block_size+1:n]);
                    }
                    
                    // if (n < heterogeneity_start_day + 15) {
                    //     a_vax_scalar = 100; 
                    //     b_vax_scalar = 2;
                    // } else 
                    
                    if (mean_vax * (1 - mean_vax) > var_vax_delta) {
                        a_vax_scalar = mean_vax * (
                            mean_vax * (1 - mean_vax) / var_vax_delta - 1
                        );
                        b_vax_scalar = (1 - mean_vax) * (
                            mean_vax * (1 - mean_vax) / var_vax_delta - 1
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
                    
                pos_counter += 1; 
                if (pos_counter == tau_vax_block_size) {
                    pos_counter = 0; 
                    pos_block += 1;
                }
            }
            
            if (include_in_omicron_wave[i][n] == 1){
                if (pos_omicron_counter == 0){    
                    if (n < tau_vax_block_size) {
                        mean_vax = ve_omicron_data[i][n];
                    } else if (include_in_omicron_wave[i][n-1] == 0) {
                        mean_vax = ve_omicron_data[i][n];
                    }else {
                        mean_vax = mean(ve_omicron_data[i][n-tau_vax_block_size+1:n]);
                    }
                    
                    if (mean_vax*(1-mean_vax) > var_vax_omicron) {
                        a_vax_scalar = mean_vax * (
                            mean_vax * (1 - mean_vax) / var_vax_omicron - 1
                        );
                        b_vax_scalar = (1 - mean_vax) * (
                            mean_vax * (1 - mean_vax) / var_vax_omicron - 1);
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
                    
                pos_omicron_counter += 1; 
                if (pos_omicron_counter == tau_vax_block_size) {
                    pos_omicron_counter = 0; 
                    pos_block_omicron += 1;
                }
            }
        }
    }
    
    // third wave model
    for (i in 1:j_third_wave){
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
        for (n in 1:N_third_wave){ 
            if (include_in_third_wave[i][n] == 1){
                idxs_third[pos_idxs] = n;
                pos_idxs += 1;  
            }
        }
        
        prop_md_third_wave[pos2_start:pos2_end] ~ beta(
            1 + count_md_third_wave[i][idxs_third[1:pos_idxs-1]], 
            1 + respond_md_third_wave[i][idxs_third[1:pos_idxs-1]]
                - count_md_third_wave[i][idxs_third[1:pos_idxs-1]]
        );
        
        // prop_masks_third_wave[pos2_start:pos2_end] ~ beta(
        //     1 + count_masks_third_wave[i][idxs_third[1:pos_idxs-1]], 
        //     1 + respond_masks_third_wave[i][idxs_third[1:pos_idxs-1]]
        //         - count_masks_third_wave[i][idxs_third[1:pos_idxs-1]]
        // );
        
        brho_third_wave[pos2_start:pos2_end] ~ beta(
            1 + imported_third_wave[idxs_third[1:pos_idxs-1],i], 
            1 + local_third_wave[idxs_third[1:pos_idxs-1],i]
        );
        
        // likelihood
        mu_hat_third_wave[pos2_start:pos2_end] ~ gamma(
            a_mu_hat_third_wave[idxs_third[1:pos_idxs-1],i], 
            b_mu_hat_third_wave[idxs_third[1:pos_idxs-1],i]
        );
    }
}

generated quantities {
    
    vector[total_N_p_third] mu_hat_third_wave_local;
    
    // third wave model
    for (i in 1:j_third_wave){
        // define these within the scope of the loop only
        int pos;
        int pos_omicron2;
        real TP_local;
        real social_measures; 
        real voc_vacc_product; 
        real susceptible_depletion_term;
        real prop_omicron_to_delta; 
        int n_omicron; 
        
        if (i == 1){
            pos = 1;
            pos_omicron2 = 1;
        } else {
            pos = pos_starts_third[i-1] + 1;
            pos_omicron2 = pos_starts_third_omicron[i-1] + 1;
        }
        
        for (n in 1:N_third_wave){
            if (include_in_third_wave[i][n] == 1){

                if (n <= omicron_start_day){
                    voc_vacc_product = voc_effect_delta * ve_delta[pos];
                } else {
                    // number of days into omicron period 
                    n_omicron = n - omicron_start_day;
                    // proportion of omicron
                    prop_omicron_to_delta = sigmoid(
                        n_omicron, tau[i], r[i], m0[i], m1[i]
                    );
                    
                    voc_vacc_product = prop_omicron_to_delta
                        * voc_effect_omicron
                        * ve_omicron[pos_omicron2]
                        + (1 - prop_omicron_to_delta)
                        * voc_effect_delta
                        * ve_delta[pos];
                        
                    pos_omicron2 += 1;
                }
                
                // social_measures = 2 * inv_logit(Mob_third_wave[i][n,:] * (bet)) 
                //     * md_third_wave[pos] 
                //     * masks_third_wave[pos];
                social_measures = 2 * inv_logit(Mob_third_wave[i][n,:] * (bet)) 
                    * md_third_wave[pos];
                    
                susceptible_depletion_term = 
                    1 - susceptible_depletion_factor * proportion_infected[n,i];
                    
                TP_local = R_Li[map_to_state_index_third[i]]
                    * voc_vacc_product
                    * social_measures;
                    
                mu_hat_third_wave_local[pos] = TP_local
                    * susceptible_depletion_term;
                
                pos += 1;
            }
        }
    }
}