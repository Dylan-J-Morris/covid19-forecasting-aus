data {
    
    int j_total;
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
    int N_third_wave;
    int j_third_wave;
    matrix[N_third_wave,j_third_wave] Reff_third_wave;
    matrix[N_third_wave,K] Mob_third_wave[j_third_wave];
    matrix[N_third_wave,K] Mob_third_wave_std[j_third_wave];
    matrix[N_third_wave,j_third_wave] sigma2_third_wave;
    vector[N_third_wave] policy_third_wave;
    matrix[N_third_wave,j_third_wave] local_third_wave;
    matrix[N_third_wave,j_third_wave] imported_third_wave;
    vector[N] count_md[j_first_wave];
    vector[N] respond_md[j_first_wave];
    vector[N_sec_wave] count_md_sec_wave[j_sec_wave];
    vector[N_sec_wave] respond_md_sec_wave[j_sec_wave];
    vector[N_third_wave] count_md_third_wave[j_third_wave];
    vector[N_third_wave] respond_md_third_wave[j_third_wave];
    vector[N_third_wave] count_masks_third_wave[j_third_wave];
    vector[N_third_wave] respond_masks_third_wave[j_third_wave];
    int map_to_state_index_first[j_first_wave];
    int map_to_state_index_sec[j_sec_wave];
    int map_to_state_index_third[j_third_wave];
    int total_N_p_sec;
    int total_N_p_third;
    vector[N] include_in_first_wave[j_first_wave];
    vector[N_sec_wave] include_in_sec_wave[j_sec_wave];
    vector[N_third_wave] include_in_third_wave[j_third_wave];
    int pos_starts_sec[j_sec_wave];
    int pos_starts_third[j_third_wave];
    int decay_start_date_third;
    vector[N_third_wave] vaccine_effect_data[j_third_wave];
    int omicron_start_day;
    vector[N_third_wave] include_in_omicron_wave[j_third_wave];
    int total_N_p_third_omicron;
    int pos_starts_third_omicron[j_third_wave];
    int total_N_p_third_omicron_3_blocks;
    int pos_starts_third_omicron_3_blocks[j_third_wave];
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
    vector[K] bet;
    real<lower=0> R_I;
    real<lower=0> R_L;
    vector<lower=0>[j_total] R_Li;
    real<lower=0> sig;
    real<lower=0> theta_md;
    matrix<lower=0,upper=1>[N,j_first_wave] prop_md;
    vector<lower=0,upper=1>[total_N_p_sec] prop_md_sec_wave;
    vector<lower=0,upper=1>[total_N_p_third] prop_md_third_wave;
    matrix<lower=0,upper=1>[N,j_first_wave] brho;
    vector<lower=0,upper=1>[total_N_p_sec] brho_sec_wave;
    vector<lower=0,upper=1>[total_N_p_third] brho_third_wave;
    real<lower=0> theta_masks;
    vector<lower=0,upper=1>[total_N_p_third] prop_masks_third_wave;
    real<lower=0> additive_voc_effect_alpha;
    real<lower=0> additive_voc_effect_delta;
    real<lower=0> additive_voc_effect_omicron;
    vector<lower=0,upper=1>[j_third_wave] eta;
    vector<lower=0>[j_third_wave] r;
    positive_ordered[N_third_wave+1] vacc_effect_ordered[j_third_wave];
    real<lower=0,upper=1> reduction_vacc_effect_omicron;
    vector<lower=0,upper=1>[total_N_p_third_omicron_3_blocks] prop_omicron_to_delta_3_day_block;
    real<lower=0,upper=1> susceptible_depletion_factor;    
}

transformed parameters {
    real<lower=0> voc_effect_alpha = 1 + additive_voc_effect_alpha;
    real<lower=0> voc_effect_delta = 1 + additive_voc_effect_delta;
    real<lower=0> voc_effect_omicron = 1 + additive_voc_effect_omicron;
    //TP models
    matrix<lower=0>[N,j_first_wave] mu_hat;
    vector<lower=0>[total_N_p_sec] mu_hat_sec_wave;
    vector<lower=0>[total_N_p_third] mu_hat_third_wave;
    //micro distancing model
    matrix<lower=0>[N,j_first_wave] md;
    vector<lower=0>[total_N_p_sec] md_sec_wave;
    vector<lower=0>[total_N_p_third] md_third_wave;
    //mask wearing model
    vector<lower=0>[total_N_p_third] masks_third_wave;
    //ordered vaccine effect with length total number of days across each jurisdiction
    vector[total_N_p_third] vacc_effect;
    vector<lower=0,upper=1>[total_N_p_third_omicron] prop_omicron_to_delta;
    //reverse the ordering of the raw vax effects in a local scope
    {
        int pos = 1;
        int pos2;
        int pos_omicron1;
        int pos_omicron2;
        int pos_omicron_counter;
        for (i in 1:j_third_wave){
            pos2 = 1;
            pos_omicron_counter = 0;
            if (i == 1){
                pos_omicron1 = 1;
                pos_omicron2 = 1;
            } else {
                pos_omicron1 = pos_starts_third_omicron[i-1]+1;
                pos_omicron2 = pos_starts_third_omicron_3_blocks[i-1]+1;
            }
            //reverse the array
            for (n in 1:N_third_wave){
                if (include_in_third_wave[i][n] == 1){
                    vacc_effect[pos] = vacc_effect_ordered[i][N_third_wave+2-pos2];
                    pos += 1;
                    pos2 += 1;
                    
                    //only want to add the component if in omicron phase
                    if (include_in_omicron_wave[i][n] == 1){
                        prop_omicron_to_delta[pos_omicron1] = prop_omicron_to_delta_3_day_block[pos_omicron2];
                        //move through the full daily vector
                        pos_omicron1 += 1;
                        pos_omicron_counter += 1;
                        if (pos_omicron_counter == 5){
                            pos_omicron_counter = 0;
                            pos_omicron2 += 1;
                        }
                    }
                }
            }
        }
    }

    //first wave model
    for (i in 1:j_first_wave) {
        real TP_local;
        real social_measures;
        for (n in 1:N){
            if (include_in_first_wave[i][n] == 1){
                md[n,i] = pow(1+theta_md, -1*prop_md[n,i]);
                social_measures = ((1-policy[n]) + md[n,i]*policy[n])*inv_logit(Mob[i][n,:]*(bet));
                TP_local = 2*R_Li[map_to_state_index_first[i]]*social_measures;
                mu_hat[n,i] = brho[n,i]*R_I + (1-brho[n,i])*TP_local;
            }
        }
    }

    //second wave model
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
                    (1-policy_sec_wave[n]) + md_sec_wave[pos]*policy_sec_wave[n]
                )*inv_logit(Mob_sec_wave[i][n,:]*(bet));
                TP_local = 2*R_Li[map_to_state_index_sec[i]]*social_measures; 
                mu_hat_sec_wave[pos] = brho_sec_wave[pos]*R_I + (1-brho_sec_wave[pos])*TP_local;
                pos += 1;
            }
        }
    }

    //third wave model
    for (i in 1:j_third_wave){
        //define these within the scope of the loop only
        int pos;
        int pos_omicron_counter;
        int pos_omicron2;
        real TP_local;
        real social_measures;
        //parameters for the vaccination effects
        real vacc_effect_tmp;
        real vacc_effect_tot;
        real decay_in_heterogeneity;
        real voc_effect_tot;
        pos_omicron_counter = 0;
        if (i == 1){
            pos = 1;
            pos_omicron2 = 1;
        } else {
            pos = pos_starts_third[i-1]+1;
            pos_omicron2 = pos_starts_third_omicron_3_blocks[i-1]+1;
        }
        for (n in 1:N_third_wave){
            if (include_in_third_wave[i][n] == 1){
                md_third_wave[pos] = pow(1+theta_md,-1*prop_md_third_wave[pos]);
                masks_third_wave[pos] = pow(1+theta_masks,-1*prop_masks_third_wave[pos]);
                //applying the return to homogeneity in vaccination effect
                if (n <= decay_start_date_third){
                    decay_in_heterogeneity = eta[i];
                } else{
                    decay_in_heterogeneity = eta[i]*exp(-r[i]*(n - decay_start_date_third));
                }
                
                //vaccine effect in the absence of Omicron
                vacc_effect_tmp = decay_in_heterogeneity + (1-decay_in_heterogeneity) * vacc_effect[pos];
                
                if (n <= omicron_start_day){
                    vacc_effect_tot = vacc_effect_tmp;
                    voc_effect_tot = voc_effect_delta;
                } else {
                    vacc_effect_tot = 1 + (
                        (prop_omicron_to_delta_3_day_block[pos_omicron2]
                        -prop_omicron_to_delta_3_day_block[pos_omicron2]*reduction_vacc_effect_omicron-1)
                        *(1-vacc_effect_tmp));
                    
                    voc_effect_tot = (
                        voc_effect_omicron * prop_omicron_to_delta_3_day_block[pos_omicron2]
                        + voc_effect_delta * (1-prop_omicron_to_delta_3_day_block[pos_omicron2]));
                    
                    pos_omicron_counter += 1;
                    if (pos_omicron_counter == 5){
                        pos_omicron2 += 1;
                        pos_omicron_counter = 0;
                    }
                }
                
                social_measures = (
                    (1-policy_third_wave[n])
                    +md_third_wave[pos]
                    *masks_third_wave[pos]
                    *policy_third_wave[n])*inv_logit(Mob_third_wave[i][n,:]*(bet));
                
                TP_local = R_Li[map_to_state_index_third[i]]*2*social_measures*voc_effect_tot*vacc_effect_tot;
                
                mu_hat_third_wave[pos] = (
                    brho_third_wave[pos]*R_I
                    + (1-brho_third_wave[pos])*TP_local)*(
                        1-susceptible_depletion_factor*cumulative_local_third[n,i]/pop_size_array[map_to_state_index_third[i]]);
                
                pos += 1;
            }
        }
    }
}

model {
    //indexers for moving through parameter vectors
    int pos2;
    int pos3;
    int pos_omicron_counter;
    int pos_omicron2;

    //fixed parameters for the vaccine effect priors
    real vacc_mu;
    real vacc_sig = 0.025;

    //priors
    //social mobility parameters
    bet ~ normal(0, 1.0);
    theta_md ~ lognormal(0, 0.5);
    theta_masks ~ lognormal(0, 0.5);

    additive_voc_effect_alpha ~ gamma(0.4*0.4/0.075, 0.4/0.075);
    additive_voc_effect_delta ~ gamma(1.5*1.5/0.05, 1.5/0.05);
    // assume that omicron is similar to delta
    additive_voc_effect_omicron ~ gamma(1.5*1.5/0.05, 1.5/0.05);

    //vaccination heterogeneity
    eta ~ beta(2, 7);
    r ~ lognormal(log(0.16), 0.1);

    //reduction in vaccine effect due to omicron
    reduction_vacc_effect_omicron ~ beta(75, 50);

    //susceptible depletion
    susceptible_depletion_factor ~ beta(2, 2);

    //hierarchical model for the baseline RL
    R_L ~ gamma(1.7*1.7/0.005,1.7/0.005);
    R_I ~ gamma(0.5*0.5/0.2,0.5/0.2);
    sig ~ exponential(1000);
    R_Li ~ gamma(R_L*R_L/sig,R_L/sig);

    //first wave model
    for (i in 1:j_first_wave) {
        for (n in 1:N){
            prop_md[n,i] ~ beta(1 + count_md[i][n], 1 + respond_md[i][n] - count_md[i][n]);
            brho[n,i] ~ beta(0.5+imported[n,i], 0.5+local[n,i]);
            mu_hat[n,i] ~ gamma(Reff[n,i]*Reff[n,i]/(sigma2[n,i]), Reff[n,i]/sigma2[n,i]);
        }
    }

    //second wave model
    for (i in 1:j_sec_wave){
        if (i == 1){
            pos2 = 1;
        } else {
            pos2 = pos_starts_sec[i-1]+1;
        }
        
        for (n in 1:N_sec_wave){
            if (include_in_sec_wave[i][n] == 1){
                prop_md_sec_wave[pos2] ~ beta(
                    1+count_md_sec_wave[i][n], 
                    1+respond_md_sec_wave[i][n]-count_md_sec_wave[i][n]);
                brho_sec_wave[pos2] ~ beta(
                    0.5+imported_sec_wave[n,i], 
                    0.5+local_sec_wave[n,i]);
                mu_hat_sec_wave[pos2] ~ gamma(
                    Reff_sec_wave[n,i]*Reff_sec_wave[n,i]/(sigma2_sec_wave[n,i]), 
                    Reff_sec_wave[n,i]/sigma2_sec_wave[n,i]);
                pos2+=1;
            }
        }
    }

    //third wave model
    for (i in 1:j_third_wave){
        pos3 = 0;
        pos_omicron_counter = 0;
        
        if (i == 1){
            pos2 = 1;
            pos_omicron2 = 1;
        } else {
            pos2 = pos_starts_third[i-1]+1;
            pos_omicron2 = pos_starts_third_omicron_3_blocks[i-1]+1;
        }
        
        for (n in 1:N_third_wave){
            if (include_in_third_wave[i][n] == 1){
                prop_md_third_wave[pos2] ~ beta(
                    1+count_md_third_wave[i][n], 
                    1+respond_md_third_wave[i][n]-count_md_third_wave[i][n]);
                prop_masks_third_wave[pos2] ~ beta(
                    1+count_masks_third_wave[i][n], 
                    1+respond_masks_third_wave[i][n]-count_masks_third_wave[i][n]);
                brho_third_wave[pos2] ~ beta(
                    0.5+imported_third_wave[n,i], 
                    0.5+local_third_wave[n,i]);
                
                if (pos3 == 0){
                    vacc_mu = vaccine_effect_data[i][n-1];
                    //for the first value assume the mean of the data as the initial value
                    vacc_effect_ordered[i][N_third_wave+1] ~ normal(vacc_mu, 0.001);
                    pos3 += 1;
                }

                vacc_mu = vaccine_effect_data[i][n];
                vacc_effect_ordered[i][N_third_wave+1-pos3] ~ normal(vacc_mu, vacc_sig);
                
                //only sample on the first day of the 3 day window assuming different prior
                if (include_in_omicron_wave[i][n] == 1){
                    if (pos_omicron_counter == 0){        
                        if (n <= omicron_start_day+20){
                            prop_omicron_to_delta_3_day_block[pos_omicron2] ~ beta(2,50);
                        } else if (n <= omicron_start_day+30){
                            prop_omicron_to_delta_3_day_block[pos_omicron2] ~ beta(2,10);
                        } else {
                            prop_omicron_to_delta_3_day_block[pos_omicron2] ~ beta(75,5);
                        }
                        pos_omicron2 += 1;
                    }
                    
                    pos_omicron_counter += 1;
                    
                    // reset the counter
                    if (pos_omicron_counter == 5){
                        pos_omicron_counter = 0;
                    }
                }
                
                mu_hat_third_wave[pos2] ~ gamma(
                    Reff_third_wave[n,i]*Reff_third_wave[n,i]/(sigma2_third_wave[n,i]), 
                    Reff_third_wave[n,i]/sigma2_third_wave[n,i]);
                
                pos2 += 1;
                pos3 += 1;
            }
        }
    }   
}