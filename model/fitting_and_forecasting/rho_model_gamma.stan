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
    vector[N_sec_wave] apply_alpha_sec_wave;                    // when to apply an increase due to alpha

    // data for the third wave  
    int N_third_wave;                                           // length of VIC days
    int j_third_wave;                                           // third wave states
    matrix[N_third_wave,j_third_wave] Reff_third_wave;          // Reff for VIC in June
    matrix[N_third_wave,K] Mob_third_wave[j_third_wave];        // Mob for VIC June
    matrix[N_third_wave,K] Mob_third_wave_std[j_third_wave];    // std of mobility
    matrix[N_third_wave,j_third_wave] sigma2_third_wave;        // variance of R_eff from previous study
    vector[N_third_wave] policy_third_wave;                     // micro distancing compliance a boolean
    matrix[N_third_wave,j_third_wave] local_third_wave;         // local cases in VIC
    matrix[N_third_wave,j_third_wave] imported_third_wave;      // imported cases in VIC

    // data relating to mobility and microdistancing
    vector[N] count_md[j_first_wave];                           // count of always
    vector[N] respond_md[j_first_wave];                         // num respondants
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
    int is_NSW[j_third_wave];                                   // indicator vector of which state is NSW in the third wave

    int decay_start_date_third;
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
    real<lower=0> additive_voc_effect_alpha;
    real<lower=0> additive_voc_effect_delta;
    real<lower=0,upper=1> eta_NSW;                              // array of adjustment factor for each third wave state
    real<lower=0,upper=1> eta_other;                            // array of adjustment factor for each third wave state
    real<lower=0> r_NSW;                                        // parameter for decay to heterogeneity
    real<lower=0> r_other;                                      // parameter for decay to heterogeneity
    vector<lower=0, upper=1>[total_N_p_third] vacc_effect;      // adjusted vaccine effect parameter centered on the supplied timeseries

}
transformed parameters {
    // this parametrisation results in voc ~ 1 + Gamma(a, b) (i.e. truncated below at 1)
    real<lower=0> voc_effect_alpha = 1 + additive_voc_effect_alpha;
    real<lower=0> voc_effect_delta = 1 + additive_voc_effect_delta;
    // TP model for each wave 
    matrix<lower=0>[N,j_first_wave] mu_hat;
    vector<lower=0>[total_N_p_sec] mu_hat_sec_wave;
    vector<lower=0>[total_N_p_third] mu_hat_third_wave;
    // microdistancing effect for each wave
    matrix<lower=0>[N,j_first_wave] md;                                    // micro distancing
    vector<lower=0>[total_N_p_sec] md_sec_wave;
    vector<lower=0>[total_N_p_third] md_third_wave;

    for (i in 1:j_first_wave) {
        real TP_local;
        real social_measures;
        for (n in 1:N){
            if (include_in_first_wave[i][n]==1) {
                md[n,i] = pow(1+theta_md , -1*prop_md[n,i]);
                social_measures = ((1-policy[n]) + md[n,i]*policy[n])*inv_logit(Mob[i][n,:]*(bet));
                //mean estimate
                TP_local = 2*R_Li[map_to_state_index_first[i]]*social_measures;
                mu_hat[n,i] = brho[n,i]*R_I + (1-brho[n,i])*TP_local; 
            }
        }
    }
    for (i in 1:j_sec_wave){
        // define these within the scope of the loop only
        int pos;
        real TP_local;
        real social_measures;
        if (i==1){
            pos=1;
        }
        else {
            //Add 1 to get to start of new group, not end of old group
            pos = pos_starts_sec[i-1]+1;
        }
        for (n in 1:N_sec_wave){
            if (include_in_sec_wave[i][n]==1){        
                md_sec_wave[pos] = pow(1+theta_md, -1*prop_md_sec_wave[pos]);
                social_measures = ((1-policy_sec_wave[n]) + md_sec_wave[pos]*policy_sec_wave[n])*inv_logit(Mob_sec_wave[i][n,:]*(bet));
                // TP_local = 2*R_Li[map_to_state_index_sec[i]]*social_measures*voc_effect_alpha; //mean estimate
                TP_local = 2*R_Li[map_to_state_index_sec[i]]*social_measures; //mean estimate
                if (apply_alpha_sec_wave[n]==1){
                    TP_local *= voc_effect_alpha;
                }
                mu_hat_sec_wave[pos] = brho_sec_wave[pos]*R_I + (1-brho_sec_wave[pos])*TP_local;
                pos += 1;
            }
        }
    }

    for (i in 1:j_third_wave){
        // define these within the scope of the loop only
        int pos;
        real TP_local;
        real social_measures;
        // parameters for the vaccination effects  
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

        for (n in 1:N_third_wave){
            if (include_in_third_wave[i][n]==1){
                md_third_wave[pos] = pow(1+theta_md, -1*prop_md_third_wave[pos]);                

                // applying the return to homogeneity in vaccination effect 
                if (n < decay_start_date_third){
                    decay_in_heterogeneity = 1.0;
                } else{
                    decay_in_heterogeneity = exp(-r*(n-decay_start_date_third));
                } 
                
                eta_tmp = eta*decay_in_heterogeneity;

                // total vaccination effect has the form of a mixture model which captures heterogeneity in the 
                // vaccination effect around the 20th of August 
                // vacc_effect_tot = eta_tmp + (1-eta_tmp) * vaccine_effect_data[i][n];
                // vacc_effect_tot = vaccine_effect_data[i][n];
                
                // instead of using the actual vaccination data as truth, use a transformed version
                vacc_effect_tot = eta_tmp + (1-eta_tmp) * max(vacc_effect[pos-1], vacc_effect[pos]);
                
                social_measures = ((1-policy_third_wave[n])+md_third_wave[pos]*policy_third_wave[n])*inv_logit(Mob_third_wave[i][n,:]*(bet));
                TP_local = 2*R_Li[map_to_state_index_third[i]]*social_measures*voc_effect_delta*vacc_effect_tot;
                
                mu_hat_third_wave[pos] = brho_third_wave[pos]*R_I + (1-brho_third_wave[pos])*TP_local;
                pos += 1;
            }
        }
    }
}
model {
    // indexer for moving through the items in the second and third waves
    int pos2;
    // shape and scale parameters for the prior distribution on the vaccination effect
    real a_vacc; 
    real b_vacc;
    // mean and variacne parameters for the beta (used for the transformation)
    real vacc_sig = 0.005;
    real vacc_mu;

    bet ~ normal(0, 1.0);
    theta_md ~ lognormal(0, 0.5);

    // note gamma parametrisation is Gamma(alpha,beta) => mean = alpha/beta 
    // parametersiing the following as voc_eff ~ 1 + gamma(a,b)
    additive_voc_effect_alpha ~ gamma(0.4*0.4/0.2, 0.4/0.2);      
    additive_voc_effect_delta ~ gamma(1.1*1.1/0.075, 1.1/0.075);
    
    // assume a hierarchical structure on the vaccine effect 
    eta_NSW ~ beta(2, 7);           // mean of 2/9
    eta_other ~ beta(2, 7);         // mean of 2/9
    // eta_NSW ~ beta(1, 100);           // mean of 2/9
    // eta_other ~ beta(1, 100);         // mean of 2/9

    // want it to have mean 0.16 => log-mean is log(0.16)
    r_NSW ~ lognormal(log(0.16), 0.1);        // r is lognormally distributed such that the mean is 28 days 
    r_other ~ lognormal(log(0.16), 0.1);        // r is lognormally distributed such that the mean is 28 days 

    R_L ~ gamma(1.8*1.8/0.005,1.8/0.005); //hyper-prior
    // R_L ~ gamma(2.0*2.0/0.005,2.0/0.005); //hyper-prior
    R_I ~ gamma(0.5*0.5/0.2,0.5/0.2);
    sig ~ exponential(50); //mean is 1/50=0.02
    R_Li ~ gamma(R_L*R_L/sig,R_L/sig); //partial pooling of state level estimates

    for (i in 1:j_first_wave) {
        for (n in 1:N){
            prop_md[n,i] ~ beta(1 + count_md[i][n], 
                                1 + respond_md[i][n] - count_md[i][n]);
            brho[n,i] ~ beta(0.5+imported[n,i], 0.5+local[n,i]); //ratio imported/ (imported + local)
            mu_hat[n,i] ~ gamma(Reff[n,i]*Reff[n,i]/(sigma2[n,i]), Reff[n,i]/sigma2[n,i]); //Stan uses shape/inverse scale
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
                prop_md_sec_wave[pos2] ~ beta(1+count_md_sec_wave[i][n], 
                                              1+respond_md_sec_wave[i][n]-count_md_sec_wave[i][n]);
                brho_sec_wave[pos2] ~ beta(0.5+imported_sec_wave[n,i], 
                                           0.5+local_sec_wave[n,i]); //ratio imported/ (imported + local)   
                mu_hat_sec_wave[pos2] ~ gamma(Reff_sec_wave[n,i]*Reff_sec_wave[n,i]/(sigma2_sec_wave[n,i]), 
                                              Reff_sec_wave[n,i]/sigma2_sec_wave[n,i]);
                pos2+=1;
            }
        }
    }

    for (i in 1:j_third_wave){
        if (i==1){
            pos2=1;
        } else {
            //Add 1 to get to start of new group, not end of old group
            pos2=pos_starts_third[i-1]+1; 
        }
        for (n in 1:N_third_wave){
            if (include_in_third_wave[i][n]==1){
                prop_md_third_wave[pos2] ~ beta(1+count_md_third_wave[i][n], 
                                                1+respond_md_third_wave[i][n]-count_md_third_wave[i][n]);
                // brho_third_wave[pos2] ~ beta(0.2+c*imported_third_wave[n,i], 
                //                              0.2+c*local_third_wave[n,i]); //ratio imported/ (imported + local)
                brho_third_wave[pos2] ~ beta(0.5+imported_third_wave[n,i], 
                                             0.5+local_third_wave[n,i]); //ratio imported/ (imported + local)
                mu_hat_third_wave[pos2] ~ gamma(Reff_third_wave[n,i]*Reff_third_wave[n,i]/(sigma2_third_wave[n,i]), 
                                                Reff_third_wave[n,i]/sigma2_third_wave[n,i]);
                
                // the mean vaccination effect should be the data supplied
                vacc_mu = vaccine_effect_data[i][n];
                // transform to shape and scale 
                a_vacc = vacc_mu*(vacc_mu*(1-vacc_mu)/vacc_sig - 1);
                b_vacc = (1-vacc_mu)*(vacc_mu*(1-vacc_mu)/vacc_sig - 1);
                // vaccine effect distributed around mean of the vaccine effect but 
                // needs to be truncated above by the previous value 
                vacc_effect[pos2] ~ beta(a_vacc, b_vacc) T[0, vaccine_effect_data[i][n-1]];
                                            
                pos2+=1;
            }
        }
    }
}