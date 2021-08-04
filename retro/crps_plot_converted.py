import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn-poster")
import seaborn as sns












crps = pd.read_csv("results/retro_big_scores.csv", header=None)
columns = ['state','data_date'] 
columns.extend([n for n in range(1,29)])
crps.columns = columns

plot_df = pd.melt(
    crps,
    id_vars = ['state','data_date']
)
plot_df['data_date'] = plot_df.data_date.apply(lambda x : x[:10])



## random walk
train = 60
rand_walk = pd.read_csv("../data/scores.csv", header=None)

columns = ['model','state','data_date'] 
columns.extend([n for n in range(1,29)])
rand_walk.columns = columns
rand_walk.head()



#skill score
def skill(model_score, base_score):
    """
    Given the model score and base score, calculate the relative score
    """
    score = 1 - model_score/base_score
    return score

rand_walk_s = rand_walk.loc[rand_walk.state.isin(states_to_plot)].set_index(['model','state','data_date'])
crps_s = crps.set_index(['state','data_date'])

score = pd.DataFrame(skill(crps_s.values, rand_walk_s.values), index=crps_s.index)

score



fig,ax = plt.subplots(figsize=(12,9), ncols=2, squeeze=False, sharey=False,sharex=True)
states_to_plot = ['NSW','VIC']

for i, state in enumerate(states_to_plot):
    row = i//2
    col = i%2
    legend=False
    if i==len(states_to_plot)-1:
        #last plot give legend
        legend = 'brief'
    sns.lineplot(
        data=plot_df.loc[plot_df.state==state],
        hue='data_date',
        x='variable',
        y='value',
        ax=ax[row,col],
        legend=legend,
    )
    ax[row,col].set_title(state)
    if col==0:
        ax[row,col].set_ylabel("CRPS")
    else:
        ax[row,col].set_ylabel("")
    if row==len(states_to_plot)//2-1:
        ax[row,col].set_xlabel("Forecast horizon")
       
       
handles, labels = ax[row,col].get_legend_handles_labels()
ax[row,col].legend(handles=handles[1:], labels=labels[1:])
plt.tight_layout()
plt.savefig("figs/retro/crps_big.png",dpi=300)






fig,ax = plt.subplots(figsize=(12,9), ncols=2, squeeze=False, sharey=True,sharex=True)
states_to_plot = ['NSW','VIC']
plot = pd.melt(score.reset_index(), id_vars =['state','data_date'])
plot.data_date = plot.data_date.apply(lambda x: x[:10])

for i, state in enumerate(states_to_plot):
    row = i//2
    col = i%2
    legend=False
    if i==len(states_to_plot)-1:
        #last plot give legend
        legend = 'brief'
    sns.lineplot(
        data=plot.loc[plot.state==state],
        x='variable',
        y='value',
        hue='data_date',
        ax=ax[row,col],
        legend=legend,
    )
    ax[row,col].set_ylim(-0.5,1)
    ax[row,col].set_yticks([0.0001],minor=True)
    ax[row,col].set_yticks([-0.5,0.5,1])
    ax[row,col].yaxis.grid(which='minor', linestyle='--',alpha=0.6, color='black')
    
    ax[row,col].set_title(state)
    if col==0:
        ax[row,col].set_ylabel("Skill score")
    else:
        ax[row,col].set_ylabel("")
    if row==len(states_to_plot)//2-1:
        ax[row,col].set_xlabel("Forecast horizon")

#remove legend title        
handles, labels = ax[row,col].get_legend_handles_labels()
ax[row,col].legend(handles=handles[1:], labels=labels[1:])
plt.savefig("figs/retro/skill_big.png",dpi=300)
plt.show()