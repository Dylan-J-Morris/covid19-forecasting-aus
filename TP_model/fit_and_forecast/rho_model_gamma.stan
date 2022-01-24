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
    vector[N_third_wave] count_masks_third_wave[j_third_wave];
    vector[N_third_wave] respond_masks_third_wave[j_third_wave];
    // vectors that map to the correct indices based on j_total 
    int map_to_state_index_first[j_first_wave];
    int map_to_state_index_sec[j_sec_wave];
    int map_to_state_index_third[j_third_wave];
    // ints for moving through the total parameter vectors in the second or third waves
    int total_N_p_sec;
    int total_N_p_third;
    // bool arrays for when to include data 
    vector[N] include_in_first_wave[j_first_wave];
    vector[N_sec_wave] include_in_sec_wave[j_sec_wave];
    vector[N_third_wave] include_in_third_wave[j_third_wave];
    // this is used to index starting points in include_in_XX_wave 
    int pos_starts_sec[j_sec_wave];
    int pos_starts_third[j_third_wave];
    // vax data
    vector[N_third_wave] vaccine_effect_data[j_third_wave];
    // data for handling omicron 
    int omicron_start_day;
    vector[N_third_wave] include_in_omicron_wave[j_third_wave];
    int total_N_p_third_omicron;
    int pos_starts_third_omicron[j_third_wave];
    int pop_size_array[j_total];    
    
}

transformed data {
    
    // for now we just calculate the cumulative number of cases in the
    // third wave as pre third wave cases were negligible
    matrix[N_third_wave,j_third_wave] cumulative_local_third;
    for (i in 1:j_third_wave){
        cumulative_local_third[:,i] = cumulative_sum(local_third_wave[:,i]);
    }
    
}

parameters {
    
    // macro and micro parameters 
    vector[K] bet;
    real<lower=0> theta_md;
    real<lower=0> theta_masks;
    matrix<lower=0,upper=1>[N,j_first_wave] prop_md;
    vector<lower=0,upper=1>[total_N_p_sec] prop_md_sec_wave;
    vector<lower=0,upper=1>[total_N_p_third] prop_md_third_wave;
    matrix<lower=0,upper=1>[N,j_first_wave] brho;
    // baseline and hierearchical RL parameters 
    real<lower=0> R_I;
    real<lower=0> R_L;
    vector<lower=0>[j_total] R_Li;
    real<lower=0> sig;
    // import parameters 
    vector<lower=0,upper=1>[total_N_p_sec] brho_sec_wave;
    vector<lower=0,upper=1>[total_N_p_third] brho_third_wave;
    vector<lower=0,upper=1>[total_N_p_third] prop_masks_third_wave;
    // voc parameters
    real<lower=0> additive_voc_effect_alpha;
    real<lower=0> additive_voc_effect_delta;
    real<lower=0> additive_voc_effect_omicron;
    // vaccine model parameters 
    vector<lower=0,upper=1>[total_N_p_third] vacc_effect;
    real<lower=0,upper=1> reduction_vacc_effect_omicron;
    vector<lower=0,upper=1>[total_N_p_third_omicron] prop_omicron_to_delta;
    real<lower=0,upper=1> susceptible_depletion_factor;    
    
}

transformed parameters {
    
    // voc parameters (shifted by 1)
    real<lower=0> voc_effect_alpha = 1 + additive_voc_effect_alpha;
    real<lower=0> voc_effect_delta = 1 + additive_voc_effect_delta;
    real<lower=0> voc_effect_omicron = 1 + additive_voc_effect_omicron;
    // TP models
    matrix<lower=0>[N,j_first_wave] mu_hat;
    vector<lower=0>[total_N_p_sec] mu_hat_sec_wave;
    vector<lower=0>[total_N_p_third] mu_hat_third_wave;
    // micro distancing model
    matrix<lower=0>[N,j_first_wave] md;
    vector<lower=0>[total_N_p_sec] md_sec_wave;
    vector<lower=0>[total_N_p_third] md_third_wave;
    // mask wearing model
    vector<lower=0>[total_N_p_third] masks_third_wave;

    // first wave model
    for (i in 1:j_first_wave) {
        real TP_local;
        real social_measures;
        
        for (n in 1:N){
            if (include_in_first_wave[i][n] == 1){
                md[n,i] = pow(1+theta_md, -1*prop_md[n,i]);
                social_measures = ((1-policy[n])+md[n,i]*policy[n])*inv_logit(Mob[i][n,:]*(bet));
                TP_local = 2*R_Li[map_to_state_index_first[i]]*social_measures;
                mu_hat[n,i] = brho[n,i]*R_I+(1-brho[n,i])*TP_local;
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
            pos = pos_starts_sec[i-1]+1;
        }
        
        for (n in 1:N_sec_wave){
            if (include_in_sec_wave[i][n] == 1){
                md_sec_wave[pos] = pow(1+theta_md, -1*prop_md_sec_wave[pos]);
                social_measures = (
                    (1-policy_sec_wave[n]) 
                    +md_sec_wave[pos]*policy_sec_wave[n])
                    *inv_logit(Mob_sec_wave[i][n,:]*(bet)
                );
                TP_local = 2*R_Li[map_to_state_index_sec[i]]*social_measures; 
                mu_hat_sec_wave[pos] = brho_sec_wave[pos]*R_I + (1-brho_sec_wave[pos])*TP_local;
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
        // parameters for the vaccination effects
        real voc_term; 
        real vacc_term; 
        real vacc_effect_omicron;
        
        if (i == 1){
            pos = 1;
            pos_omicron2 = 1;
        } else {
            pos = pos_starts_third[i-1]+1;
            pos_omicron2 = pos_starts_third_omicron[i-1]+1;
        }
        
        for (n in 1:N_third_wave){
            if (include_in_third_wave[i][n] == 1){
                md_third_wave[pos] = pow(1+theta_md, -1*prop_md_third_wave[pos]);
                masks_third_wave[pos] = pow(1+theta_masks, -1*prop_masks_third_wave[pos]);

                if (n <= omicron_start_day){
                    voc_term = voc_effect_delta;
                    vacc_term = vacc_effect[pos];
                } else {
                    vacc_effect_omicron = 1 - reduction_vacc_effect_omicron*(1-vacc_effect[pos]);
                    voc_term = prop_omicron_to_delta[pos_omicron2]*voc_effect_omicron 
                        + (1-prop_omicron_to_delta[pos_omicron2])*voc_effect_delta;
                    vacc_term = prop_omicron_to_delta[pos_omicron2]*vacc_effect_omicron 
                        + (1-prop_omicron_to_delta[pos_omicron2])*vacc_effect[pos];
                    pos_omicron2 += 1;
                }
                
                social_measures = (
                    (1-policy_third_wave[n])
                    +md_third_wave[pos]
                    *masks_third_wave[pos]
                    *policy_third_wave[n])
                    *inv_logit(Mob_third_wave[i][n,:]*(bet));

                TP_local = R_Li[map_to_state_index_third[i]]*2*social_measures*voc_term*vacc_term;
                
                mu_hat_third_wave[pos] = (
                    (brho_third_wave[pos]*R_I+(1-brho_third_wave[pos])*TP_local)*(
                    1-susceptible_depletion_factor*cumulative_local_third[n,i]/pop_size_array[map_to_state_index_third[i]])
                );
                    
                pos += 1;
            }
        }
    }
}

model {

    // indexers for moving through parameter vectors
    int pos2_start;
    int pos2_end;
    int pos2;
    real pos_omicron_counter;   // this needs to be real for correct floating point division 
    int pos_omicron2;
    int pos_omicron2_start;
    int pos_omicron2_end;

    // fixed parameters for the vaccine effect priors
    real mean_vax;
    real var_vax = 0.0005;
    array[N_third_wave] real a_vax;
    array[N_third_wave] real b_vax;
    
    // drifting omicron proportion to 0.9
    real drift_mean_omicron = 0.8;
    real drift_factor;
    real mean_omicron; 
    real var_omicron = 0.005; 
    array[N_third_wave] real a_omicron;
    array[N_third_wave] real b_omicron;
    int days_third_left = 0;
    array[N_sec_wave] int idxs_sec;
    array[N_third_wave] int idxs_third;
    array[N_third_wave] int idxs_third_omicron;
    int pos_idxs;
    int pos_omicron_idxs;

    // priors
    // social mobility parameters
    bet ~ normal(0, 1.0);
    theta_md ~ lognormal(0, 0.5);
    theta_masks ~ lognormal(0, 0.5);
    
    // gives full priors of 1 + Gamma() for each VoC effect
    additive_voc_effect_alpha ~ gamma(0.4*0.4/0.075, 0.4/0.075);
    additive_voc_effect_delta ~ gamma(1.5*1.5/0.05, 1.5/0.05);
    additive_voc_effect_omicron ~ gamma(1.5*1.5/0.05, 1.5/0.05);

    // reduction in vaccine effect due to omicron
    reduction_vacc_effect_omicron ~ beta(40, 60);   //mean of 0.4 - slightly lower than supplied VE ts 

    // susceptible depletion
    susceptible_depletion_factor ~ beta(2, 2);

    // hierarchical model for the baseline RL
    R_L ~ gamma(1.7*1.7/0.005,1.7/0.005);
    R_I ~ gamma(0.5*0.5/0.2,0.5/0.2);
    sig ~ exponential(250);
    R_Li ~ gamma(R_L*R_L/sig,R_L/sig);

    // first wave model
    for (i in 1:j_first_wave) { 
        prop_md[:,i] ~ beta(1+count_md[i][:], 1+respond_md[i][:]-count_md[i][:]);
        brho[:,i] ~ beta(0.5+imported[:,i], 0.5+local[:,i]);
        // likelihood
        mu_hat[:,i] ~ gamma(Reff[:,i] .* Reff[:,i] ./ (sigma2[:,i]), Reff[:,i] ./ sigma2[:,i]);
    }

    // second wave model
    for (i in 1:j_sec_wave){
        pos_idxs = 1;
        if (i == 1){
            pos2_start = 1;
            pos2_end = pos_starts_sec[i];
        } else {
            pos2_start = pos_starts_sec[i-1]+1;
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
            1+count_md_sec_wave[i][idxs_sec[1:pos_idxs-1]], 
            1+respond_md_sec_wave[i][idxs_sec[1:pos_idxs-1]]-count_md_sec_wave[i][idxs_sec[1:pos_idxs-1]]
        );
        brho_sec_wave[pos2_start:pos2_end] ~ beta(
            0.5+imported_sec_wave[idxs_sec[1:pos_idxs-1],i], 
            0.5+local_sec_wave[idxs_sec[1:pos_idxs-1],i]
        );

        // likelihood
        mu_hat_sec_wave[pos2_start:pos2_end] ~ gamma(
            Reff_sec_wave[idxs_sec[1:pos_idxs-1],i] .* Reff_sec_wave[idxs_sec[1:pos_idxs-1],i] ./ (sigma2_sec_wave[idxs_sec[1:pos_idxs-1],i]), 
            Reff_sec_wave[idxs_sec[1:pos_idxs-1],i] ./ sigma2_sec_wave[idxs_sec[1:pos_idxs-1],i]
        );
    }
    
    // third wave model
    for (i in 1:j_third_wave){
        pos_omicron_counter = 0;
        days_third_left = 0;
        pos_idxs = 1;
        pos_omicron_idxs = 1;
        
        if (i == 1){
            pos2_start = 1;
            pos2_end = pos_starts_third[i];
            pos_omicron2_start = 1;
            pos_omicron2_end = pos_starts_third_omicron[i];
        } else {
            pos2_start = pos_starts_third[i-1]+1;
            pos2_end = pos_starts_third[i];
            pos_omicron2_start = pos_starts_third_omicron[i-1]+1;
            pos_omicron2_end = pos_starts_third_omicron[i];
        }
        
        // create an array for indexing the proportion terms
        for (n in 1:N_third_wave){ 
            if (include_in_third_wave[i][n] == 1){
                idxs_third[pos_idxs] = n;
                pos_idxs += 1;  
            }
        }
        
        prop_md_third_wave[pos2_start:pos2_end] ~ beta(
            1+count_md_third_wave[i][idxs_third[1:pos_idxs-1]], 
            1+respond_md_third_wave[i][idxs_third[1:pos_idxs-1]]-count_md_third_wave[i][idxs_third[1:pos_idxs-1]]
        );
        prop_masks_third_wave[pos2_start:pos2_end] ~ beta(
            1+count_masks_third_wave[i][idxs_third[1:pos_idxs-1]], 
            1+respond_masks_third_wave[i][idxs_third[1:pos_idxs-1]]-count_masks_third_wave[i][idxs_third[1:pos_idxs-1]]
        );
        brho_third_wave[pos2_start:pos2_end] ~ beta(
            0.5+imported_third_wave[idxs_third[1:pos_idxs-1],i], 
            0.5+local_third_wave[idxs_third[1:pos_idxs-1],i]
        );
        
        // calculate the shape and scale for each n 
        for (n in 1:pos_idxs-1){
            // vaccine effect has mean the supplied estimate
            mean_vax = vaccine_effect_data[i][idxs_third[n]];
            if (mean_vax*(1-mean_vax) > var_vax) {
                a_vax[n] = mean_vax*(mean_vax*(1-mean_vax)/var_vax - 1);
                b_vax[n] = (1-mean_vax)*(mean_vax*(1-mean_vax)/var_vax - 1);
            } else if (mean_vax > 0.98) {
                a_vax[n] = 50;
                b_vax[n] = 2;
            } else if (mean_vax < 0.02) {
                a_vax[n] = 2; 
                b_vax[n] = 50;
            } else {
                a_vax[n] = 1; 
                b_vax[n] = 1;
            }
        }
        
        // sample the vaccination effect 
        vacc_effect[pos2_start:pos2_end] ~ beta(a_vax[1:pos_idxs-1], b_vax[1:pos_idxs-1]);
        
        for (n in 1:N_third_wave){
            if ((include_in_third_wave[i][n] == 1) && (include_in_omicron_wave[i][n] == 1)){
                if (days_third_left == 0){
                    days_third_left = N_third_wave - n;
                }
                
                if (n <= omicron_start_day+15 || pos_omicron_counter <= 3){
                    // sample differently pre December 1st or in the first 5 days
                    a_omicron[pos_omicron_idxs] = 2;
                    b_omicron[pos_omicron_idxs] = 50;
                } else {
                    drift_factor = pos_omicron_counter / days_third_left;
                    // // mean is a mixture of the mean proportion for last few days and the drift to 0.95
                    mean_omicron = (1-drift_factor)*mean(prop_omicron_to_delta[(pos_omicron_idxs-3):(pos_omicron_idxs-1)]) 
                        + drift_factor*drift_mean_omicron;
                    
                    if (mean_omicron*(1-mean_omicron) > var_omicron){
                        // calculate shape and scale parameters
                        a_omicron[pos_omicron_idxs] = mean_omicron*(mean_omicron*(1-mean_omicron)/var_omicron - 1);
                        b_omicron[pos_omicron_idxs] = (1-mean_omicron)*(mean_omicron*(1-mean_omicron)/var_omicron - 1);
                    } else if (mean_omicron > 0.98) {
                        // if we are close to 1 or reduced assume a slightly different prior
                        a_omicron[pos_omicron_idxs] = 50;
                        b_omicron[pos_omicron_idxs] = 2;
                    } else if (mean_omicron < 0.02) {
                        a_omicron[pos_omicron_idxs] = 2;
                        b_omicron[pos_omicron_idxs] = 50;
                    } else {
                        a_omicron[pos_omicron_idxs] = 1; 
                        b_omicron[pos_omicron_idxs] = 1;
                    }
                }
                pos_omicron_counter += 1.0;
                pos_omicron_idxs += 1;
                
                pos2 += 1;
            }
        }
        
        prop_omicron_to_delta[pos_omicron2_start:pos_omicron2_end] ~ beta(a_omicron[1:pos_omicron_idxs-1], b_omicron[1:pos_omicron_idxs-1]);     
        
        // likelihood
        mu_hat_third_wave[pos2_start:pos2_end] ~ gamma(
            Reff_third_wave[idxs_third[1:pos_idxs-1],i] .* Reff_third_wave[idxs_third[1:pos_idxs-1],i] ./ (sigma2_third_wave[idxs_third[1:pos_idxs-1],i]), 
            Reff_third_wave[idxs_third[1:pos_idxs-1],i] ./ sigma2_third_wave[idxs_third[1:pos_idxs-1],i]
        );
    }
}