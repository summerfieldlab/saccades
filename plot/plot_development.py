""" Visualising qualitative comparisons with human behaviour. Order of class mastery and coherence effect."""
#%%
import os
import numpy as np
from scipy.stats import ttest_ind
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from itertools import product
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.linear_model import LinearRegression
import ghibli
import utils
sns.set(font_scale=2)

# The first dimension of the confusion matrices indexes the true numerosity and
# the second indexes the predicted numerosity. So the upper triangle represent
# overestimations and the lower triangle represent underestimations.


#%% Simple Counting
df = pd.DataFrame()
min_num = 1
moving_avg_n = 25
threshold = 0.95
#%%
dfs = []
for rep in range(50, 55):
    try:
        results_dir = '../results/logpolar/'
        base_name = f'bigcnn_hsize-1024_input-shape_noise_num1-5_nl-0.74_policy-cheat+jitter_trainshapes-BCDEsame__gw6_12_100000_loss-num_opt-Adam_drop0.5_sort_count-notA_10eps_rep{rep}'
        ffile = f'{results_dir}/batch_confusion_{base_name}.npy'
        batch_conf = np.load(ffile)
        print(batch_conf.shape)
        cnn_reshaped = batch_conf.reshape(-1,5,5)
        class_wise_perf_cnn = [np.convolve(cnn_reshaped[:,i,i], np.ones(moving_avg_n)/moving_avg_n, mode='valid') for i in range(5)]
        numerosities = [1, 2, 3, 4, 5]
        for idx, num in enumerate(numerosities):
            perf = class_wise_perf_cnn[idx]
            dfs.append(pd.DataFrame({'Smoothed Accuracy':perf, 'n updates':range(len(perf)), 'number class':num, 'Model':'CNN', }))
            suprathreshold = np.where(perf > threshold)[0]
            if len(suprathreshold) > 0:
                mb = suprathreshold[0]
                df = pd.concat([df, pd.DataFrame([{'N':num, 'updates to threshold':mb, 'model':'CNN',
                                                   'Task':'Simple Counting','rep':rep}])], ignore_index=True)
            else:
                print(f'bigcnn did not train long enough to reach threshold on class {num}.')
    except: 
        print(f'missing {base_name}')

    try:
        # base_name = 'bigcnn_hsize-1024_input-shape_noise_num1-5_nl-0.74_policy-cheat+jitter_trainshapes-BCDEsame__gw6_12_100000_loss-num_opt-Adam_drop0.5_sort_count-notA_10eps_rep50'
        base_name = f'rnn_classifier2stream_hsize-1024_input-both_logpolar_num1-5_nl-0.74_policy-cheat+jitter_trainshapes-BCDEsame__logpolar_12_100000_loss-both_opt-Adam_drop0.5_sort_count-notA_100eps_rep{rep}'
        ffile = f'{results_dir}/batch_confusion_{base_name}.npy'
        batch_conf = np.load(ffile)
        dual_reshaped = batch_conf.reshape(-1,5,5)
        class_wise_perf_dual = [np.convolve(dual_reshaped[:,i,i], np.ones(moving_avg_n)/moving_avg_n, mode='valid') for i in range(5)]
        numerosities = [1, 2, 3, 4, 5]
        for idx, num in enumerate(numerosities):
            perf = class_wise_perf_dual[idx]
            dfs.append(pd.DataFrame({'Smoothed Accuracy':perf, 'n updates':range(len(perf)), 'number class':num, 'Model':'Dual-stream'}))
            suprathreshold = np.where(perf > threshold)[0]
            if len(suprathreshold) > 0:
                mb = suprathreshold[0]
                df = pd.concat([df, pd.DataFrame([{'N':num, 'updates to threshold':mb, 'model':'Dual-Stream', 
                                                   'Task':'Simple Counting','rep':rep}])], ignore_index=True)
            else:
                print(f'dual-stream did not train long enough to reach threshold on class {num}. ')
    except:
        print(f'missing {base_name}')
df_acc_simp = pd.concat(dfs, axis=0)    
#%%
ghibli.set_palette('PonyoMedium3')
sns.lineplot(data=df, x='N', y='updates to threshold', hue='model', errorbar='sd')
plt.ylabel(f'Updates to {threshold}% \ntraining accuracy')
plt.xticks([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
plt.title('Simple Counting')
#%%
fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
gfg = sns.lineplot(data=df_acc_simp[df_acc_simp['Model']=='CNN'], y='Smoothed Accuracy', x='n updates', hue='number class', errorbar='sd', ax=ax1)
ax1.set_title('CNN')
ax1.set_ylabel('Smoothed training \n accuracy (%)')
ax1.set_xlabel('Updates')
ax1.set_yticks([0, 0.5, 1], [0, 50, 100])
sns.lineplot(data=df_acc_simp[df_acc_simp['Model']=='Dual-stream'], y='Smoothed Accuracy', x='n updates', hue='number class', errorbar='sd', ax=ax2, legend=False)
ax2.set_title('Dual-stream RNN')
ax2.set_xlim([0, 10000])
ax2.set_xlabel('Updates')
gfg.legend(fontsize=15)
plt.tight_layout()
plt.savefig('smoothed_acc_per_class.png', dpi=150, bbox_inches='tight')


#%% Ignore Distractors
# df = pd.DataFrame()
min_num = 1
# threshold = 0.95 # like in developmental studies with "Give N" task
dfs = []
for rep in range(50, 54):
    try:
        results_dir = '../results/logpolar/'
        base_name = f'bigcnn_hsize-1024_input-shape_noise_num1-5_nl-0.74_policy-cheat+jitter_trainshapes-BCDEsame_distract012_gw6_12_100000_loss-num_opt-Adam_drop0.5_sort_count-notA_10eps_rep{rep}'
        ffile = f'{results_dir}/batch_confusion_{base_name}.npy'
        batch_conf = np.load(ffile).mean(axis=2) # Average over different number of distractors
        print(batch_conf.shape)
        cnn_reshaped = batch_conf.reshape(-1,5,5)
        class_wise_perf_cnn = [np.convolve(cnn_reshaped[:,i,i], np.ones(moving_avg_n)/moving_avg_n, mode='valid') for i in range(5)]
        numerosities = [1, 2, 3, 4, 5]
        for idx, num in enumerate(numerosities):
            perf = class_wise_perf_cnn[idx]
            dfs.append(pd.DataFrame({'Smoothed Accuracy':perf, 'n updates':range(len(perf)), 'number class':num, 'Model':'CNN'}))
            suprathreshold = np.where(perf > threshold)[0]
            if len(suprathreshold) > 0:
                mb = suprathreshold[0]
                df = pd.concat([df, pd.DataFrame([{'N':num, 'updates to threshold':mb, 
                                                   'model':'CNN', 'Task':'Ignore Distractors','rep':rep}])], ignore_index=True)
            else:
                print(f'bigcnn did not train long enough to reach threshold on class {num}. ')
    except: 
        print(f'missing {base_name}')

    try:
        # base_name = 'bigcnn_hsize-1024_input-shape_noise_num1-5_nl-0.74_policy-cheat+jitter_trainshapes-BCDEsame__gw6_12_100000_loss-num_opt-Adam_drop0.5_sort_count-notA_10eps_rep50'
        # base_name = f'pretrained_ventral-cnn-mse_hsize-1024_input-both_logpolar_num1-5_nl-0.74_policy-cheat+jitter_trainshapes-BCDEsame_distract012_logpolar_12_100000_loss-both_opt-Adam_drop0.5_sort_count-notA_100eps_rep{rep}'
        base_name = f'pretrained_ventral-cnn-mse_hsize-1024_input-both_logpolar_num1-5_nl-0.74_policy-cheat+jitter_trainshapes-BCDEsame_distract012_logpolar_12_100000_loss-both_opt-Adam_drop0.5_sort_count-notA_300eps_rep{rep}'
        ffile = f'{results_dir}/batch_confusion_{base_name}.npy'
        batch_conf = np.load(ffile).mean(axis=2) # Average over different number of distractors
        print(batch_conf.shape)
        dual_reshaped = batch_conf.reshape(-1,5,5)
        class_wise_perf_dual = [np.convolve(dual_reshaped[:,i,i], np.ones(moving_avg_n)/moving_avg_n, mode='valid') for i in range(5)]
        numerosities = [1, 2, 3, 4, 5]
        # df_acc_simple = pd.Dataframe(columns=['Smoothed Accuracy', 'n updates', 'number class'])
        
        for idx, num in enumerate(numerosities):
            perf = class_wise_perf_dual[idx]
            dfs.append(pd.DataFrame({'Smoothed Accuracy':perf, 'n updates':range(len(perf)), 'number class':num, 'Model':'Dual-stream'}))
            suprathreshold = np.where(perf > threshold)[0]
            if len(suprathreshold) > 0:
                mb = suprathreshold[0]
                df = pd.concat([df, pd.DataFrame([{'N':num, 'updates to threshold':mb, 
                                                   'model':'Dual-Stream', 'Task':'Ignore Distractors','rep':rep}])], ignore_index=True)
            else:
                print(f'dual-stream did not train long enough to reach threshold on class {num}. ')
        
    except:
        print(f'missing {base_name}')
df_acc_ign = pd.concat(dfs, axis=0)
        
#%%
ghibli.set_palette('PonyoMedium3')
# sns.lineplot(data=df, x='N', y='updates to threshold', hue='model', errorbar='sd')
sns.catplot(data=df, y='updates to threshold', x='N', col='model', hue='Task', kind='box', errorbar='sd', sharey=False)
# plt.ylabel(f'Updates to {threshold}% \ntraining accuracy')
plt.xticks([0, 1, 2, 3, 4], [1, 2, 3, 4, 5])
# plt.title('Ignore Distractors')
# %%
conf = 'rnn_classifier2stream_hsize-1024_input-both_logpolar_num1-5_nl-0.74_policy-cheat+jitter_trainshapes-BCDEsame__logpolar_12_100000_loss-both_opt-Adam_drop0.5_sort_count-notA_20eps_rep52'
ffile = f'{results_dir}/confusion_{conf}.npy'
conf = np.load(ffile)

# %%
# sns.lineplot(data=df_acc_ign, y='Smoothed Accuracy', x='n updates', hue='number class', errorbar='sd')
fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
sns.lineplot(data=df_acc_ign[df_acc_ign['Model']=='CNN'], y='Smoothed Accuracy', x='n updates', hue='number class', errorbar='sd', ax=ax1, legend=False)
sns.lineplot(data=df_acc_ign[df_acc_ign['Model']=='Dual-stream'], y='Smoothed Accuracy', x='n updates', hue='number class', errorbar='sd', ax=ax2)
# %%
# ghibli.set_palette('MarnieMedium1')
sns.set_palette('magma')
for i in range(5):
    plt.plot(class_wise_perf_dual[i])
    # plt.plot(dual_reshaped[:,i,i])
    # plt.plot(class_wise_perf_cnn[i])
    # plt.plot(dual_reshaped[:,i,i])
plt.xlim([0,9000])
plt.legend(range(1,6))
plt.ylabel('Smoothed Accuracy')
plt.title('Simple Counting')
plt.xlabel('# updates')

# %% Coherence Illusion
results_dir = '../results/logpolar/'
base_name = f'rnn_classifier2stream_hsize-1024_input-both_logpolar_num1-5_nl-0.74_policy-cheat+jitter_trainshapes-BCDEmixed__logpolar_12_100000_loss-both_opt-Adam_drop0.5_sort_count-notA_300eps_rep'


# diff_conf = conf[-1, 0, :, :]
# same_conf = conf[-1, 1, :, :]

# fig, (ax1, ax2) = plt.subplots(1, 2)
# ax1.matshow(diff_conf)
# ax2.matshow(same_conf)

entries = []
# add loop for rep
for rep in range(20):
    
    try:
        ffile = f'{results_dir}/confusion_{base_name}{rep}.npy'
        conf = np.load(ffile)
        print(rep)
    except:
        print(f'no rep {rep}')
    for ep, mat in enumerate(conf):
        # [diff1_loader, diff06_loader, diff03_loader, same_loader]
        diff_conf = mat[0, :, :]
        diff06_conf = mat[1, :, :]
        diff03_conf = mat[2, :, :]
        same_conf = mat[3, :, :]

        diffover = diff_conf[np.triu_indices_from(diff_conf, k=1)].sum()
        diffunder = diff_conf[np.tril_indices_from(diff_conf, k=-1)].sum()
        diff06over = diff06_conf[np.triu_indices_from(diff06_conf, k=1)].sum()
        diff06under = diff06_conf[np.tril_indices_from(diff06_conf, k=-1)].sum()
        diff03over = diff03_conf[np.triu_indices_from(diff03_conf, k=1)].sum()
        diff03under = diff03_conf[np.tril_indices_from(diff03_conf, k=-1)].sum()
        sameover = same_conf[np.triu_indices_from(same_conf, k=1)].sum()
        sameunder = same_conf[np.tril_indices_from(same_conf, k=-1)].sum()
        entries.append({'Item_Coherence':0, 'Over-Under': diffover-diffunder, 'Epoch':ep})
        entries.append({'Item_Coherence':2/4, 'Over-Under': diff06over-diff06under, 'Epoch':ep})
        entries.append({'Item_Coherence':3/4, 'Over-Under': diff03over-diff03under, 'Epoch':ep})
        entries.append({'Item_Coherence':1, 'Over-Under': sameover-sameunder, 'Epoch':ep})
        
    
df = pd.DataFrame(entries)
#%%
sns.lineplot(data=df[df.Epoch>100], y='Over-Under', x='Epoch', hue='Item_Coherence')
plt.title('Coherence Illusion Test')
plt.ylabel('Number of overestimations minus \n number of underestimations')
# %%
fig, ax = plt.subplots()
last_epoch = df[df.Epoch==300]
sns.boxplot(data=last_epoch, y='Over-Under', x='Item_Coherence', ax=ax)
sns.stripplot(data=last_epoch, y='Over-Under', x='Item_Coherence', ax=ax, dodge=True, alpha=0.6)
ticks = [-40, -20, 0, 20, 40]
# ax.set_yticks(ticks, labels=[t/50 for t in ticks])
ax.set_ylabel('Number of overestimations minus \n number of underestimations \n (out of 5000 test examples)')
# ax.set_title('Coherence illusion test')
ax.set_xlabel('Item coherence \n(Number of unique item shapes)')
ax.set_xticklabels([' 4', '3', '2', '1 '])
# plt.tight_layout()
# %% Do stats
a = last_epoch.query('Item_Coherence==1')['Over-Under'].values
b = last_epoch.query('Item_Coherence==0.75')['Over-Under'].values
c = last_epoch.query('Item_Coherence==0.5')['Over-Under'].values
d = last_epoch.query('Item_Coherence==0.0')['Over-Under'].values
result = ttest_ind(a, b, alternative='greater')
print(result)
# Ttest_indResult(statistic=17.253604046391043, pvalue=7.178093135595223e-20)
result = ttest_ind(b, c, alternative='greater')
print(result)
# Ttest_indResult(statistic=4.271798317988012, pvalue=6.253079668633553e-05)
result = ttest_ind(c, d, alternative='greater')
print(result)
# Ttest_indResult(statistic=-0.9471999284110761, pvalue=0.8252391204942917)
# %%
