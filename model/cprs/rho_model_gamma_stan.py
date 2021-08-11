import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sys import argv
import pystan
import os, glob
from Reff_functions import *
from Reff_constants import *

from params import infer_vacc_effect

## Define pystan model
rho_model_gamma1 = """
data {
    // data for the initial wave
    int N;                                                      //data length num days
    int K;                                                      //Number of mobility indices
    int j;                                                      //Number of states
    matrix[N,j] Reff;                                           //response
    matrix[N,K] Mob[j];                                         //Mobility indices
    matrix[N,K] Mob_std[j];                                     //std of mobility
    matrix[N,j] sigma2;                                         //Variances of R_eff from previous study
    vector[N] policy;                                           //Indicators for post policy or not
    matrix[N,j] local;                                          //local number of cases
    matrix[N,j] imported;                                       //imported number of cases

    // data for the secondary wave
    int N_sec_wave;                                             //length of VIC days
    int j_sec_wave;                                             //second wave states
    matrix[N_sec_wave,j_sec_wave] Reff_sec_wave;                //Reff for VIC in June
    matrix[N_sec_wave,K] Mob_sec_wave[j_sec_wave];              //Mob for VIC June
    matrix[N_sec_wave,K] Mob_sec_wave_std[j_sec_wave];          // std of mobility
    matrix[N_sec_wave,j_sec_wave] sigma2_sec_wave;              // variance of R_eff from previous study
    vector[N_sec_wave] policy_sec_wave;                         // micro distancing compliance
    matrix[N_sec_wave,j_sec_wave] local_sec_wave;               //local cases in VIC
    matrix[N_sec_wave,j_sec_wave] imported_sec_wave;            //imported cases in VIC

    // data for the third wave -- 
    int N_third_wave;                                           //length of VIC days
    int j_third_wave;                                           //thirdond wave states
    matrix[N_third_wave,j_third_wave] Reff_third_wave;          //Reff for VIC in June
    matrix[N_third_wave,K] Mob_third_wave[j_third_wave];        //Mob for VIC June
    matrix[N_third_wave,K] Mob_third_wave_std[j_third_wave];    // std of mobility
    matrix[N_third_wave,j_third_wave] sigma2_third_wave;        // variance of R_eff from previous study
    vector[N_third_wave] policy_third_wave;                     // micro distancing compliance
    matrix[N_third_wave,j_third_wave] local_third_wave;         //local cases in VIC
    matrix[N_third_wave,j_third_wave] imported_third_wave;      //imported cases in VIC

    // data relating to mobility and microdistancing
    vector[N] count_md[j];                                      //count of always
    vector[N] respond_md[j];                                    // num respondants
    vector[N_sec_wave] count_md_sec_wave[j_sec_wave];           //count of always
    vector[N_sec_wave] respond_md_sec_wave[j_sec_wave];         // num respondants
    vector[N_third_wave] count_md_third_wave[j_third_wave];     //count of always
    vector[N_third_wave] respond_md_third_wave[j_third_wave];   // num respondants

    int map_to_state_index_sec[j_sec_wave];                     // indices of second wave to map to first
    int map_to_state_index_third[j_third_wave];                 // indices of second wave to map to first
    int total_N_p_sec;                                          //total number of data in sec wave, entire state first
    int total_N_p_third;                                        //total number of data in sec wave, entire state first
    vector[N_sec_wave] include_in_sec_wave[j_sec_wave];         // dates include in sec_wave 
    vector[N_third_wave] include_in_third_wave[j_third_wave];   // dates include in sec_wave 
    int pos_starts_sec[j_sec_wave];                             //starting positions for each state in the second wave
    int pos_starts_third[j_third_wave];                         //starting positions for each state in the third wave 
    vector[N_third_wave] vaccination_data[j_third_wave];        //vaccination data

}
parameters {
    vector[K] bet;                                              //coefficients
    real<lower=0> R_I;                                          //base level imports,
    real<lower=0> R_L;                                          //base level local
    vector<lower=0>[j] R_Li;                                    //state level estimates
    real<lower=0> sig;                                          //state level variance
    real<lower=0> theta_md;                                     // md weighting
    matrix<lower=0,upper=1>[N,j] prop_md;                       // proportion who are md'ing
    vector<lower=0,upper=1>[total_N_p_sec] prop_md_sec_wave;
    vector<lower=0,upper=1>[total_N_p_third] prop_md_third_wave;
    matrix<lower=0,upper=1>[N,j] brho;                          //estimate of proportion of imported cases
    matrix<lower=0,upper=1>[N,K] noise[j];
    //real<lower=0> R_temp;

    vector<lower=0,upper=1>[total_N_p_sec] brho_sec_wave;       //estimate of proportion of imported cases
    vector<lower=0,upper=1>[total_N_p_third] brho_third_wave;   //estimate of proportion of imported cases

    // voc and vaccine effects
    real<lower=0> VoC_effect_third_wave_0;
    real<lower=0> sig_VoC_effect_third_wave;
    real<lower=0> VoC_effect_third_wave;

    real<lower=0> sig_vaccine_effect_third_wave;
    real<lower=0> vaccine_effect_third_wave;
}
transformed parameters {
    matrix<lower=0>[N,j] mu_hat;
    vector<lower=0>[total_N_p_sec] mu_hat_sec_wave;
    vector<lower=0>[total_N_p_third] mu_hat_third_wave;
    matrix<lower=0>[N,j] md;                                    //micro distancing
    vector<lower=0>[total_N_p_sec] md_sec_wave;
    vector<lower=0>[total_N_p_third] md_third_wave;

    for (i in 1:j) {
        for (n in 1:N){
            md[n,i] = pow(1+theta_md , -1*prop_md[n,i]);
            //mean estimate
            mu_hat[n,i] = brho[n,i]*R_I + (1-brho[n,i])*2*R_Li[i]*(
            (1-policy[n]) + md[n,i]*policy[n] )*inv_logit(
            Mob[i][n,:]*(bet)); 
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
                if (map_to_state_index_sec[i] == 5) {
                    mu_hat_sec_wave[pos] = brho_sec_wave[pos]*R_I + (1-brho_sec_wave[pos])*(2*R_Li[
                    map_to_state_index_sec[i]
                    ])*(
                    (1-policy_sec_wave[n]) + md_sec_wave[pos]*policy_sec_wave[n] )*inv_logit(
                    Mob_sec_wave[i][n,:]*(bet)); //mean estimate
                }
                else {
                    mu_hat_sec_wave[pos] = brho_sec_wave[pos]*R_I + (1-brho_sec_wave[pos])*2*R_Li[
                    map_to_state_index_sec[i]
                    ]*(
                    (1-policy_sec_wave[n]) + md_sec_wave[pos]*policy_sec_wave[n] )*inv_logit(
                    Mob_sec_wave[i][n,:]*(bet)); //mean estimate
                }
                pos += 1;
            }
        }
    }
    for (i in 1:j_third_wave){
        // define these within the scope of the loop only
        int pos;
        real vacc_effect;
        if (i==1){
            pos=1;
        }
        else {
            //Add 1 to get to start of new group, not end of old group
            pos =pos_starts_third[i-1]+1;
            }
        for (n in 1:N_third_wave){
            if (include_in_third_wave[i][n]==1){
                md_third_wave[pos] = pow(1+theta_md ,-1*prop_md_third_wave[pos]);
                if (map_to_state_index_third[i] == 5) {

                    vacc_effect = vaccine_effect_third_wave * vaccination_data[i][n];
                    
                    mu_hat_third_wave[pos] = brho_third_wave[pos]*R_I + (1-brho_third_wave[pos])*(2*R_Li[
                    map_to_state_index_third[i]
                    ])*(
                    (1-policy_third_wave[n]) + md_third_wave[pos]*policy_third_wave[n] )*inv_logit(
                    Mob_third_wave[i][n,:]*(bet)) * VoC_effect_third_wave * vacc_effect; //mean estimate
                }
                else {

                    vacc_effect = vaccine_effect_third_wave * vaccination_data[i][n];

                    mu_hat_third_wave[pos] = brho_third_wave[pos]*R_I + (1-brho_third_wave[pos])*2*R_Li[
                    map_to_state_index_third[i]
                    ]*(
                    (1-policy_third_wave[n]) + md_third_wave[pos]*policy_third_wave[n] )*inv_logit(
                    Mob_third_wave[i][n,:]*(bet))* VoC_effect_third_wave * vacc_effect; //mean estimate
                }
                pos += 1;
            }
        }
    }

}
model {
    int pos2;
    real voc_mean;
    bet ~ normal(0,1);
    theta_md ~ lognormal(0,0.5);

    //note gamma parametrisation is Gamma(alpha,beta) => mean = alpha/beta 
    voc_mean = 2.0;
    VoC_effect_third_wave_0 ~ gamma(voc_mean*voc_mean/0.02,voc_mean/0.02);
    sig_VoC_effect_third_wave ~ exponential(20);
    VoC_effect_third_wave ~ gamma(VoC_effect_third_wave_0*VoC_effect_third_wave_0/sig_VoC_effect_third_wave, 
                                  VoC_effect_third_wave_0/sig_VoC_effect_third_wave);

    sig_vaccine_effect_third_wave ~ exponential(50);
    vaccine_effect_third_wave ~ normal(1, sig_vaccine_effect_third_wave);

    R_L ~ gamma(1.8*1.8/0.05,1.8/0.05); //hyper-prior
    R_I ~ gamma(0.5*0.5/.2,0.5/.2);
    sig ~ exponential(20); //mean is 1/50=0.02
    R_Li ~ gamma(R_L*R_L/sig, R_L/sig); //partial pooling of state level estimates

    for (i in 1:j) {
        for (n in 1:N){
            prop_md[n,i] ~ beta(1 + count_md[i][n], 1+ respond_md[i][n] - count_md[i][n]);
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
                //noise_sec_wave[i][n,:] ~ normal( Mob_sec_wave[i][n,:] , Mob_sec_wave_std[i][n,:]);
                mu_hat_sec_wave[pos2] ~ gamma( Reff_sec_wave[n,i]*Reff_sec_wave[n,i]/(sigma2_sec_wave[n,i]), Reff_sec_wave[n,i]/sigma2_sec_wave[n,i]);
                pos2+=1;
            }
        }
    }

    for (i in 1:j_third_wave){
        if (i==1){
            pos2=1;
        }
        else {
            //Add 1 to get to start of new group, not end of old group
            pos2 =pos_starts_third[i-1]+1; 
            }
        for (n in 1:N_third_wave){
            if (include_in_third_wave[i][n]==1){
                prop_md_third_wave[pos2] ~ beta(1 + count_md_third_wave[i][n], 1+ respond_md_third_wave[i][n] - count_md_third_wave[i][n]);
                brho_third_wave[pos2] ~ beta( 1+ imported_third_wave[n,i], 1+ local_third_wave[n,i]); //ratio imported/ (imported + local)
                //noise_third_wave[i][n,:] ~ normal( Mob_third_wave[i][n,:] , Mob_third_wave_std[i][n,:]);
                mu_hat_third_wave[pos2] ~ gamma(Reff_third_wave[n,i]*Reff_third_wave[n,i]/(sigma2_third_wave[n,i]), Reff_third_wave[n,i]/sigma2_third_wave[n,i]);
                pos2+=1;
            }
        }
    }
}
"""

rho_model_gamma2 = """
data {
    // data for the initial wave
    int N;                                                      //data length num days
    int K;                                                      //Number of mobility indices
    int j;                                                      //Number of states
    matrix[N,j] Reff;                                           //response
    matrix[N,K] Mob[j];                                         //Mobility indices
    matrix[N,K] Mob_std[j];                                     //std of mobility
    matrix[N,j] sigma2;                                         //Variances of R_eff from previous study
    vector[N] policy;                                           //Indicators for post policy or not
    matrix[N,j] local;                                          //local number of cases
    matrix[N,j] imported;                                       //imported number of cases

    // data for the secondary wave
    int N_sec_wave;                                             //length of VIC days
    int j_sec_wave;                                             //second wave states
    matrix[N_sec_wave,j_sec_wave] Reff_sec_wave;                //Reff for VIC in June
    matrix[N_sec_wave,K] Mob_sec_wave[j_sec_wave];              //Mob for VIC June
    matrix[N_sec_wave,K] Mob_sec_wave_std[j_sec_wave];          // std of mobility
    matrix[N_sec_wave,j_sec_wave] sigma2_sec_wave;              // variance of R_eff from previous study
    vector[N_sec_wave] policy_sec_wave;                         // micro distancing compliance
    matrix[N_sec_wave,j_sec_wave] local_sec_wave;               //local cases in VIC
    matrix[N_sec_wave,j_sec_wave] imported_sec_wave;            //imported cases in VIC

    // data for the third wave -- 
    int N_third_wave;                                           //length of VIC days
    int j_third_wave;                                           //thirdond wave states
    matrix[N_third_wave,j_third_wave] Reff_third_wave;          //Reff for VIC in June
    matrix[N_third_wave,K] Mob_third_wave[j_third_wave];        //Mob for VIC June
    matrix[N_third_wave,K] Mob_third_wave_std[j_third_wave];    // std of mobility
    matrix[N_third_wave,j_third_wave] sigma2_third_wave;        // variance of R_eff from previous study
    vector[N_third_wave] policy_third_wave;                     // micro distancing compliance
    matrix[N_third_wave,j_third_wave] local_third_wave;         //local cases in VIC
    matrix[N_third_wave,j_third_wave] imported_third_wave;      //imported cases in VIC

    // data relating to mobility and microdistancing
    vector[N] count_md[j];                                      //count of always
    vector[N] respond_md[j];                                    // num respondants
    vector[N_sec_wave] count_md_sec_wave[j_sec_wave];           //count of always
    vector[N_sec_wave] respond_md_sec_wave[j_sec_wave];         // num respondants
    vector[N_third_wave] count_md_third_wave[j_third_wave];     //count of always
    vector[N_third_wave] respond_md_third_wave[j_third_wave];   // num respondants

    int map_to_state_index_sec[j_sec_wave];                     // indices of second wave to map to first
    int map_to_state_index_third[j_third_wave];                 // indices of second wave to map to first
    int total_N_p_sec;                                          //total number of data in sec wave, entire state first
    int total_N_p_third;                                        //total number of data in sec wave, entire state first
    vector[N_sec_wave] include_in_sec_wave[j_sec_wave];         // dates include in sec_wave 
    vector[N_third_wave] include_in_third_wave[j_third_wave];   // dates include in sec_wave 
    int pos_starts_sec[j_sec_wave];                             //starting positions for each state in the second wave
    int pos_starts_third[j_third_wave];                         //starting positions for each state in the third wave 
    vector[N_third_wave] vaccination_data[j_third_wave];        //vaccination data

}
parameters {
    vector[K] bet;                                              //coefficients
    real<lower=0> R_I;                                          //base level imports,
    real<lower=0> R_L;                                          //base level local
    vector<lower=0>[j] R_Li;                                    //state level estimates
    real<lower=0> sig;                                          //state level variance
    real<lower=0> theta_md;                                     // md weighting
    matrix<lower=0,upper=1>[N,j] prop_md;                       // proportion who are md'ing
    vector<lower=0,upper=1>[total_N_p_sec] prop_md_sec_wave;
    vector<lower=0,upper=1>[total_N_p_third] prop_md_third_wave;
    matrix<lower=0,upper=1>[N,j] brho;                          //estimate of proportion of imported cases
    matrix<lower=0,upper=1>[N,K] noise[j];
    //real<lower=0> R_temp;

    vector<lower=0,upper=1>[total_N_p_sec] brho_sec_wave;       //estimate of proportion of imported cases
    vector<lower=0,upper=1>[total_N_p_third] brho_third_wave;   //estimate of proportion of imported cases

    // voc and vaccine effects
    real<lower=0> VoC_effect_third_wave_0;
    real<lower=0> sig_VoC_effect_third_wave;
    real<lower=0> VoC_effect_third_wave;

}
transformed parameters {
    matrix<lower=0>[N,j] mu_hat;
    vector<lower=0>[total_N_p_sec] mu_hat_sec_wave;
    vector<lower=0>[total_N_p_third] mu_hat_third_wave;
    matrix<lower=0>[N,j] md;                                    //micro distancing
    vector<lower=0>[total_N_p_sec] md_sec_wave;
    vector<lower=0>[total_N_p_third] md_third_wave;

    for (i in 1:j) {
        for (n in 1:N){
            md[n,i] = pow(1+theta_md , -1*prop_md[n,i]);
            //mean estimate
            mu_hat[n,i] = brho[n,i]*R_I + (1-brho[n,i])*2*R_Li[i]*(
            (1-policy[n]) + md[n,i]*policy[n] )*inv_logit(
            Mob[i][n,:]*(bet)); 
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
                if (map_to_state_index_sec[i] == 5) {
                    mu_hat_sec_wave[pos] = brho_sec_wave[pos]*R_I + (1-brho_sec_wave[pos])*(2*R_Li[
                    map_to_state_index_sec[i]
                    ])*(
                    (1-policy_sec_wave[n]) + md_sec_wave[pos]*policy_sec_wave[n] )*inv_logit(
                    Mob_sec_wave[i][n,:]*(bet)); //mean estimate
                }
                else {
                    mu_hat_sec_wave[pos] = brho_sec_wave[pos]*R_I + (1-brho_sec_wave[pos])*2*R_Li[
                    map_to_state_index_sec[i]
                    ]*(
                    (1-policy_sec_wave[n]) + md_sec_wave[pos]*policy_sec_wave[n] )*inv_logit(
                    Mob_sec_wave[i][n,:]*(bet)); //mean estimate
                }
                pos += 1;
            }
        }
    }
    for (i in 1:j_third_wave){
        // define these within the scope of the loop only
        int pos;
        real vacc_effect;
        if (i==1){
            pos=1;
        }
        else {
            //Add 1 to get to start of new group, not end of old group
            pos =pos_starts_third[i-1]+1;
            }
        for (n in 1:N_third_wave){
            if (include_in_third_wave[i][n]==1){
                md_third_wave[pos] = pow(1+theta_md ,-1*prop_md_third_wave[pos]);
                if (map_to_state_index_third[i] == 5) {
                    
                    mu_hat_third_wave[pos] = brho_third_wave[pos]*R_I + (1-brho_third_wave[pos])*(2*R_Li[
                    map_to_state_index_third[i]
                    ])*(
                    (1-policy_third_wave[n]) + md_third_wave[pos]*policy_third_wave[n] )*inv_logit(
                    Mob_third_wave[i][n,:]*(bet)) * VoC_effect_third_wave; //mean estimate
                }
                else {

                    mu_hat_third_wave[pos] = brho_third_wave[pos]*R_I + (1-brho_third_wave[pos])*2*R_Li[
                    map_to_state_index_third[i]
                    ]*(
                    (1-policy_third_wave[n]) + md_third_wave[pos]*policy_third_wave[n] )*inv_logit(
                    Mob_third_wave[i][n,:]*(bet))* VoC_effect_third_wave; //mean estimate
                }
                pos += 1;
            }
        }
    }

}
model {
    int pos2;
    real voc_mean;
    bet ~ normal(0,1);
    theta_md ~ lognormal(0,0.5);

    //note gamma parametrisation is Gamma(alpha,beta) => mean = alpha/beta 
    voc_mean = 2.0;
    VoC_effect_third_wave_0 ~ gamma(voc_mean*voc_mean/0.02,voc_mean/0.02);
    sig_VoC_effect_third_wave ~ exponential(20);
    VoC_effect_third_wave ~ gamma(VoC_effect_third_wave_0*VoC_effect_third_wave_0/sig_VoC_effect_third_wave, 
                                  VoC_effect_third_wave_0/sig_VoC_effect_third_wave);

    R_L ~ gamma(1.8*1.8/0.05,1.8/0.05); //hyper-prior
    R_I ~ gamma(0.5*0.5/.2,0.5/.2);
    sig ~ exponential(20); //mean is 1/50=0.02
    R_Li ~ gamma(R_L*R_L/sig, R_L/sig); //partial pooling of state level estimates

    for (i in 1:j) {
        for (n in 1:N){
            prop_md[n,i] ~ beta(1 + count_md[i][n], 1+ respond_md[i][n] - count_md[i][n]);
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
                //noise_sec_wave[i][n,:] ~ normal( Mob_sec_wave[i][n,:] , Mob_sec_wave_std[i][n,:]);
                mu_hat_sec_wave[pos2] ~ gamma( Reff_sec_wave[n,i]*Reff_sec_wave[n,i]/(sigma2_sec_wave[n,i]), Reff_sec_wave[n,i]/sigma2_sec_wave[n,i]);
                pos2+=1;
            }
        }
    }

    for (i in 1:j_third_wave){
        if (i==1){
            pos2=1;
        }
        else {
            //Add 1 to get to start of new group, not end of old group
            pos2 =pos_starts_third[i-1]+1; 
            }
        for (n in 1:N_third_wave){
            if (include_in_third_wave[i][n]==1){
                prop_md_third_wave[pos2] ~ beta(1 + count_md_third_wave[i][n], 1+ respond_md_third_wave[i][n] - count_md_third_wave[i][n]);
                brho_third_wave[pos2] ~ beta( 1+ imported_third_wave[n,i], 1+ local_third_wave[n,i]); //ratio imported/ (imported + local)
                //noise_third_wave[i][n,:] ~ normal( Mob_third_wave[i][n,:] , Mob_third_wave_std[i][n,:]);
                mu_hat_third_wave[pos2] ~ gamma(Reff_third_wave[n,i]*Reff_third_wave[n,i]/(sigma2_third_wave[n,i]), Reff_third_wave[n,i]/sigma2_third_wave[n,i]);
                pos2+=1;
            }
        }
    }
}
"""

# regardless of whether we infer the vacc effect we still need to pass data to stan but this will 
# most likely be deprecated at some time in the future
if infer_vacc_effect:
    rho_model_gamma = rho_model_gamma1
else:
    rho_model_gamma = rho_model_gamma2