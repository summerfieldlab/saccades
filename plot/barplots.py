"""Load results and prepare plots for paper figures."""
#%%
import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
# from sklearn.metrics.pairwise import euclidean_distances
from sklearn.linear_model import LinearRegression
from scipy.stats import mannwhitneyu, wilcoxon
import ghibli
import plot_utils

# sns.set_palette('colorblind')
sns.set(font_scale=2)

# Create custom colorblind-friendly palettes
colorblind = sns.color_palette('colorblind').as_hex()
gray = colorblind[7]
pastel = sns.color_palette("pastel").as_hex()
light_gray = pastel[7]
colors = [gray, light_gray]
colors.extend( colorblind[0:7])
colors2 = [gray, light_gray]
colors2.extend(colorblind[2:7])

# Set your custom color palette
# sns.set_palette(sns.color_palette(colors))
# sns.set_palette(ghibli.ghibli_palettes['PonyoMedium'])
ghibli.set_palette('PonyoMedium')
# sns.set_color_codes(palette='colorblind')
# sns.set_palette("colorblind")

home = '/home/jessica/Dropbox/saccades/rnn_tests/'
eye_home1 = '/home/jessica/Dropbox/saccades/eye_tracking/tim/'
eye_home2 = '/home/jessica/Dropbox/saccades/eye_tracking/julian/'
res_dir = home + 'results/logpolar'
letters_dir = home + 'results/toy/letters'
# scale_dir = 'results/toy/scaled'
model_desc_temp = '{}_hsize-1024_input-{}'
data_desc_temp = '{}_num1-5_nl-0.74_policy-{}_trainshapes-{}same_{}_{}_{}_100000'
train_desc_temp = 'loss-{}_opt-Adam_drop0.5_sort_count-{}_300eps'
# confusion_pretrained_ventral-cnn-mse_hsize-1024_input-both_logpolar_num1-5_nl-0.74_policy-cheat+jitter_trainshapes-BCDEsame_distract012_logpolar_12_100000_loss-both_opt-Adam_drop0.5_sort_count-multi_300eps_rep0
fs = 20
nreps = 30
figsize=(5.5, 4)

# ******************************************************* FIGURE 2 ************
#%% CNN baseline on both tasks
model_desc = model_desc_temp.format('bigcnn', 'shape')
data_desc = data_desc_temp.format('noise', 'cheat+jitter', 'BCDE', '', 'gw6', 12)
# te_data_desc = data_desc_temp.format('both', '5000')
train_desc = train_desc_temp.format('num', 'multi')

cnn_simp_nomap = plot_utils.get_results(nreps, model_desc, data_desc, train_desc)
cnn_simp_nomap = cnn_simp_nomap.query('epoch==300')
cnn_simp_nomap['Map loss'] = False
cnn_simp_nomap['Task'] = 'Simple \nCounting'
cnn_simp_nomap['Model'] = 'CNN'
cnn_simp_nomap['Dataset'] = cnn_simp_nomap.apply(lambda x: plot_utils.relabel_dataset(x.dataset, x['test shapes'],
                                                                                     x['test lums']), axis=1)
sim_val = cnn_simp_nomap.query('Dataset=="Validation"')['accuracy count'].values
sim_both = cnn_simp_nomap.query('Dataset=="OOD Both"')['accuracy count'].values
mean_diff = sim_val-sim_both
print(mean_diff)
print(f'CNN simple mean diff = {mean_diff.mean()}+/- {np.std(mean_diff)}')

data_desc = data_desc_temp.format('noise', 'cheat+jitter', 'BCDE', 'distract012', 'gw6', 12)
cnn_dist_nomap = plot_utils.get_results(nreps, model_desc, data_desc, train_desc)
cnn_dist_nomap = cnn_dist_nomap.query('epoch==300')
cnn_dist_nomap['Map loss'] = False
cnn_dist_nomap['Task'] = 'Ignore \nDistractors'
cnn_dist_nomap['Dataset'] = cnn_dist_nomap.apply(lambda x: plot_utils.relabel_dataset(x.dataset, x['test shapes'],
                                                                                 x['test lums']), axis=1)
cnn_dist_nomap['Model'] = 'CNN'
dist_val = cnn_dist_nomap.query('Dataset=="Validation"')['accuracy count'].values
dist_both = cnn_dist_nomap.query('Dataset=="OOD Both"')['accuracy count'].values
mean_diff = dist_val-dist_both
print(mean_diff)
print(f'CNN distractor mean diff = {mean_diff.mean()}+/- {np.std(mean_diff)}')
cnn_nomap = pd.concat((cnn_simp_nomap, cnn_dist_nomap))
# cnn_nomap.groupby(['Dataset', 'Task']).agg(Mean=('accuracy count', np.mean),
                                        #    Std=('accuracy count', np.std))
# 20 hits for bigcnn_hsize-1024_input-shape
# [18.62 63.4  27.92 25.1  27.16 26.86 20.82 45.3  47.9  55.64 32.82 24.4
#  56.74 38.54 35.2  46.28 23.96 41.6  27.38 53.96]
# CNN simple mean diff = 36.980000000000004+/- 13.199313618518197
# 23 hits for bigcnn_hsize-1024_input-shape
# [73.7  72.96 76.22 74.06 76.44 75.68 75.2  74.4  71.5  72.2  77.1  76.68
#  73.98 74.32 75.98 73.84 75.72 74.44 77.36 74.   71.12 74.08 71.78]
# CNN distractor mean diff = 74.46782608695652+/- 1.7376619887496993
# %% Test whether OOD performance is significantly worse than validation
stat, p = wilcoxon(cnn_simp_nomap.query('Dataset=="Validation"')['accuracy count'].values,
                   cnn_simp_nomap.query('Dataset=="OOD Shape"')['accuracy count'].values,
                   alternative='greater')
print(f'w={stat}, Bonferonni corrected p={p*9}')
stat, p = wilcoxon(cnn_simp_nomap.query('Dataset=="Validation"')['accuracy count'].values,
                   cnn_simp_nomap.query('Dataset=="OOD Luminance"')['accuracy count'].values,
                   alternative='greater')
print(f'w={stat}, Bonferonni corrected p={p*9}')
stat, p = wilcoxon(cnn_simp_nomap.query('Dataset=="Validation"')['accuracy count'].values,
                   cnn_simp_nomap.query('Dataset=="OOD Both"')['accuracy count'].values,
                   alternative='greater')
print(f'w={stat}, Bonferonni corrected p={p*9}')
#%% plot
ghibli.set_palette('PonyoMedium')
sns.barplot(data=cnn_nomap, y='accuracy count', x='Task', hue='Dataset', legend=True)
# sns.swarmplot(data=cnn_nomap, y='accuracy count', x='Task', hue='Dataset', dodge=True, 
            #   size=2, edgecolor='black', linewidth=0.5, alpha=0.5, legend=False)
ghibli.set_palette('PonyoLight')
# ghibli.set_palette('PonyoMedium')
# ghibli.set_palette('PonyoDark')
sns.stripplot(data=cnn_nomap, y='accuracy count', x='Task', hue='Dataset', dodge=True, 
              size=5, edgecolor='black', linewidth=1, alpha=0.5, legend=False)
plt.plot([-0.5, 1.5], [20, 20], '--', color='gray', label='Chance')
# sns.boxplot(data=cnn_nomap, y='accuracy count', x='task', hue='dataset')
# sns.stripplot(data=cnn_nomap, y='accuracy count', x='task', hue='dataset')
plt.title('CNN')
plt.ylabel('Accuracy (%)', )
plt.grid('on')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
plt.savefig('CNN_accuracy.png', dpi=300, bbox_inches='tight')


# %% DUAL STREAM MODEL
model_desc = model_desc_temp.format('rnn_classifier2stream', 'both')
data_desc = data_desc_temp.format('logpolar', 'cheat+jitter', 'BCDE', '', 'logpolar', 12)
train_desc = train_desc_temp.format('both', 'multi')
dual_simp_map = plot_utils.get_results(nreps, model_desc, data_desc, train_desc)
dual_simp_map = dual_simp_map.query('epoch==300')
dual_simp_map['Map loss'] = True
dual_simp_map['Task'] = 'Simple \nCounting'
dual_simp_map['Dataset'] = dual_simp_map.apply(lambda x: plot_utils.relabel_dataset(x.dataset, x['test shapes'], x['test lums']), axis=1)
dual_simp_map['Model'] = 'Dual-stream RNN'
sim_val = dual_simp_map.query('Dataset=="Validation"')['accuracy count'].values
sim_both = dual_simp_map.query('Dataset=="OOD Both"')['accuracy count'].values
mean_diff = sim_val-sim_both
print(mean_diff)
print(f'Dual stream simple mean diff = {mean_diff.mean()}+/- {np.std(mean_diff)}')

model_desc = model_desc_temp.format('pretrained_ventral-cnn-mse', 'both')
data_desc = data_desc_temp.format('logpolar', 'cheat+jitter', 'BCDE', 'distract012', 'logpolar', 12)
dual_dist_map = plot_utils.get_results(nreps, model_desc, data_desc, train_desc)
dual_dist_map = dual_dist_map.query('epoch==300')
dual_dist_map['Map loss'] = True
dual_dist_map['Task'] = 'Ignore \nDistractors'
dual_dist_map['Dataset'] = dual_simp_map.apply(lambda x: plot_utils.relabel_dataset(x.dataset, x['test shapes'], x['test lums']), axis=1)
dual_dist_map['Model'] = 'Dual-stream RNN'
dist_val = dual_dist_map.query('Dataset=="Validation"')['accuracy count'].values
dist_both = dual_dist_map.query('Dataset=="OOD Both"')['accuracy count'].values
mean_diff = dist_val-dist_both
print(mean_diff)
print(f'Dual stream distractor mean diff = {mean_diff.mean()}+/- {np.std(mean_diff)}')
dual_map = pd.concat((dual_simp_map, dual_dist_map))
simp = pd.concat([dual_simp_map, cnn_simp_nomap])
# dual_map.groupby(['Dataset', 'Task']).agg(Mean=('accuracy count', np.mean),
                                        #    Std=('accuracy count', np.std))
# 21 hits for rnn_classifier2stream_hsize-1024_input-both
# [1.78 1.86 1.9  2.12 1.84 1.72 1.74 1.86 2.4  1.96 1.98 1.84 1.3  1.72
#  2.2  1.56 1.88 1.82 2.36 1.54 1.76]
# Dual stream simple mean diff = 1.8638095238095245+/- 0.2515944618679638
# 21 hits for pretrained_ventral-cnn-mse_hsize-1024_input-both
# [0.98 0.22 0.8  0.46 0.86 0.94 1.02 1.14 0.7  0.82 0.2  0.94 0.84 0.98
#  0.82 0.48 0.98 0.6  1.04 0.84 0.74]
# Dual stream distractor mean diff = 0.7809523809523792+/- 0.251280620250308                                  
#%%
stat, p = wilcoxon(dual_simp_map.query('Dataset=="Validation"')['accuracy count'].values,
             dual_simp_map.query('Dataset=="OOD Shape"')['accuracy count'].values,
             alternative='greater')
print(f'w={stat}, Bonferonni corrected p={p*9}')
stat, p = wilcoxon(dual_simp_map.query('Dataset=="Validation"')['accuracy count'].values,
             dual_simp_map.query('Dataset=="OOD Luminance"')['accuracy count'].values,
             alternative='greater')
print(f'w={stat}, Bonferonni corrected p={p*9}')
stat, p = wilcoxon(dual_simp_map.query('Dataset=="Validation"')['accuracy count'].values,
             dual_simp_map.query('Dataset=="OOD Both"')['accuracy count'].values,
             alternative='greater')
print(f'w={stat}, Bonferonni corrected p={p*9}')


stat, p = wilcoxon(dual_dist_map.query('Dataset=="Validation"')['accuracy count'].values,
             dual_dist_map.query('Dataset=="OOD Shape"')['accuracy count'].values,
             alternative='greater')
print(f'w={stat}, Bonferonni corrected p={p*9}')
stat, p = wilcoxon(dual_dist_map.query('Dataset=="Validation"')['accuracy count'].values,
             dual_dist_map.query('Dataset=="OOD Luminance"')['accuracy count'].values,
             alternative='greater')
print(f'w={stat}, Bonferonni corrected p={p*9}')
stat, p = wilcoxon(dual_dist_map.query('Dataset=="Validation"')['accuracy count'].values,
             dual_dist_map.query('Dataset=="OOD Both"')['accuracy count'].values,
             alternative='greater')
print(f'w={stat}, Bonferonni corrected p={p*9}')

# %% plot
ghibli.set_palette('PonyoMedium')
sns.barplot(data=dual_map, y='accuracy count', x='Task', hue='Dataset')
# sns.swarmplot(data=dual_map, y='accuracy count', x='Task', hue='Dataset', dodge=True, 
#               size=4, edgecolor='black', linewidth=0.5, alpha=0.5, legend=False)
ghibli.set_palette('PonyoLight')
sns.stripplot(data=dual_map, y='accuracy count', x='Task', hue='Dataset', dodge=True, 
              size=5, edgecolor='black', linewidth=1, alpha=0.5, legend=False)
plt.plot([-0.5, 1.5], [20, 20], '--', color='gray', label='Chance')
plt.legend(fontsize=15)
plt.title('Dual Stream RNN')
plt.ylabel('Accuracy (%)')
plt.grid('on')

# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
plt.savefig('dual-stream_accuracy.png', dpi=300, bbox_inches='tight')

# %% Stats comparing CNN and dual-stream
no_tests = 9
stat_df = pd.DataFrame()
i=0
for dataset in ['OOD Shape', 'OOD Luminance', 'OOD Both']:
    x = dual_simp_map.query('Dataset==@dataset')['accuracy count'].values
    y = cnn_dist_nomap.query('Dataset==@dataset')['accuracy count'].values
    U1, p = mannwhitneyu(x, y, alternative='greater')
    U2 = x.shape[0] * y.shape[0] - U1
    stat = U1 #min(U1, U2)
    dic = {'Task':'Simple Counting', 'Dataset':dataset, 'U1':U1, 'U2':U2, 
           'U':stat, 'Bonferonni corrected p': p*no_tests}
    stat_df = pd.concat((stat_df, pd.DataFrame(dic, index=[i])))
    i+=1
    
    x = dual_dist_map.query('Dataset==@dataset')['accuracy count'].values
    y = cnn_dist_nomap.query('Dataset==@dataset')['accuracy count'].values
    U1, p = mannwhitneyu(x, y, alternative='greater')
    U2 = x.shape[0] * y.shape[0] - U1
    stat = U1 #min(U1, U2)
    dic = {'Task':'Ignore Distractors', 'Dataset':dataset, 'U1':U1, 'U2':U2, 
           'U':stat, 'Bonferonni corrected p': p*no_tests}
    stat_df = pd.concat((stat_df, pd.DataFrame(dic, index=[i])))
    i+=1

## ********************************************** EXTRA FIGURES ON MAP LOSS ***
# %% CNN with map
model_desc = model_desc_temp.format('bigcnn', 'shape')
data_desc = data_desc_temp.format('noise', 'cheat+jitter', 'BCDE', '', 'gw6', 12)
train_desc = train_desc_temp.format('both', 'multi')
cnn_simp_map = plot_utils.get_results(nreps, model_desc, data_desc, train_desc)
cnn_simp_map = cnn_simp_map.query('epoch==300')
cnn_simp_map['Dataset'] = cnn_simp_map.apply(lambda x: plot_utils.relabel_dataset(x.dataset, x['test shapes'], x['test lums']), axis=1)
cnn_simp_map['Map loss'] = True
cnn_simp_map['Model'] = 'CNN+map'
cnn_simp_map['Task'] = 'Simple \nCounting'
cnn_simp = pd.concat((cnn_simp_nomap, cnn_simp_map))


# %% plot
ghibli.set_palette('PonyoMedium')
sns.barplot(data=cnn_simp, y='accuracy count', hue='Dataset', x='Map loss')
sns.stripplot(data=cnn_simp, y='accuracy count', hue='Dataset', x='Map loss', dodge=True, 
              size=3, edgecolor='black', linewidth=0.5, alpha=0.4, legend=False)
plt.plot([-0.5, 1.5], [20, 20], '--', color='gray', label='Chance')
plt.title('CNN on Simple Counting')
plt.ylabel('Accuracy (%)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
plt.savefig('CNN_simple_map.png', dpi=300, bbox_inches='tight')
# %% CNN IGNORE DISTRACTORS with map
model_desc = model_desc_temp.format('bigcnn', 'shape')
data_desc = data_desc_temp.format('noise', 'cheat+jitter', 'BCDE', 'distract012', 'gw6', 12)
train_desc = train_desc_temp.format('both', 'multi')
cnn_dist_map = plot_utils.get_results(nreps, model_desc, data_desc, train_desc)
cnn_dist_map = cnn_dist_map.query('epoch==300')
cnn_dist_map['Dataset'] = cnn_dist_map.apply(lambda x: plot_utils.relabel_dataset(x.dataset, x['test shapes'], x['test lums']), axis=1)
cnn_dist_map['Map loss'] = True
cnn_dist_map['Model'] = 'CNN+map'
cnn_dist_map['Task'] = 'Ignore \nDistractors'
cnn_dist = pd.concat((cnn_dist_nomap, cnn_dist_map))

cnn_map = pd.concat((cnn_simp_map, cnn_dist_map))
cnn_map.groupby(['Dataset', 'Task']).agg(Mean=('accuracy count', np.mean),
                                           Std=('accuracy count', np.std))
# %% plot
sns.barplot(data=cnn_dist, y='accuracy count', hue='Dataset', x='Map loss')
sns.stripplot(data=cnn_dist, y='accuracy count', hue='Dataset', x='Map loss', dodge=True,
              size=3, edgecolor='black', linewidth=0.5, alpha=0.4, legend=False)
plt.plot([-0.5, 1.5], [20, 20], '--', color='gray', label='Chance')
plt.title('CNN on Ignore Distractors')
plt.ylabel('Accuracy (%)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
plt.savefig('CNN_dist_map.png', dpi=300, bbox_inches='tight')

# %% Dual Stream No Map
# Simple
model_desc = model_desc_temp.format('rnn_classifier2stream', 'both')
data_desc = data_desc_temp.format('logpolar', 'cheat+jitter', 'BCDE', '', 'logpolar', 12)
# te_data_desc = data_desc_temp.format('both', '5000')
train_desc = train_desc_temp.format('num', 'multi')
dual_simp_nomap = plot_utils.get_results(nreps, model_desc, data_desc, train_desc)
train_desc = train_desc_temp.format('num', 'notA')
dual_simp_nomap2 = plot_utils.get_results(nreps, model_desc, data_desc, train_desc)
if isinstance(dual_simp_nomap2, pd.DataFrame):
    dual_simp_nomap = pd.concat((dual_simp_nomap, dual_simp_nomap2))
dual_simp_nomap = dual_simp_nomap.query('epoch==300')
dual_simp_nomap['Map loss'] = False
dual_simp_nomap['Task'] = 'Simple \nCounting'
dual_simp_nomap['Dataset'] = dual_simp_nomap.apply(lambda x: plot_utils.relabel_dataset(x.dataset, x['test shapes'], x['test lums']), axis=1)
dual_simp_nomap['Model'] = 'Dual-stream no map'
dual_simp = pd.concat((dual_simp_map, dual_simp_nomap))
# %% plot
sns.barplot(data=dual_simp, y='accuracy count', hue='Dataset', x='Map loss')
sns.stripplot(data=dual_simp, y='accuracy count', hue='Dataset', x='Map loss', dodge=True, 
              size=3, edgecolor='black', linewidth=0.5, alpha=0.4, legend=False)
plt.plot([-0.5, 1.5], [20, 20], '--', color='gray', label='Chance')
plt.title('Dual Stream on Simple Counting')
plt.ylabel('Accuracy (%)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
plt.savefig('dualstream_simple_map.png', dpi=300, bbox_inches='tight')
#%% Ignore Distractors
model_desc = model_desc_temp.format('pretrained_ventral-cnn-mse', 'both')
data_desc = data_desc_temp.format('logpolar', 'cheat+jitter', 'BCDE', 'distract012', 'logpolar', 12)
train_desc = train_desc_temp.format('num', 'multi')
dual_dist_nomap = plot_utils.get_results(nreps, model_desc, data_desc, train_desc)
train_desc = train_desc_temp.format('num', 'notA')
dual_dist_nomap2 = plot_utils.get_results(nreps, model_desc, data_desc, train_desc)
if isinstance(dual_dist_nomap2, pd.DataFrame):
    dual_dist_nomap = pd.concat((dual_dist_nomap, dual_dist_nomap2))
dual_dist_nomap = dual_dist_nomap.query('epoch==300')
dual_dist_nomap['Map loss'] = False
dual_dist_nomap['Task'] = 'Ignore \nDistractors'
dual_dist_nomap['Dataset'] = dual_dist_nomap.apply(lambda x: plot_utils.relabel_dataset(x.dataset, x['test shapes'], x['test lums']), axis=1)
dual_dist_nomap['Model'] = 'Dual-stream no map'
dual_dist = pd.concat((dual_dist_map, dual_dist_nomap))

dual_nomap = pd.concat((dual_simp_nomap, dual_dist_nomap))
dual_nomap.groupby(['Dataset', 'Task']).agg(Mean=('accuracy count', np.mean),
                                           Std=('accuracy count', np.std))
# %% plot
# sns.set_palette("colorblind")
# sns.set_palette("Set3")
# sns.set_palette("Dark2")
# sns.set_palette("Accent")
# sns.set_palette("Pastel2")

sns.barplot(data=dual_dist, y='accuracy count', hue='Dataset', x='Map loss')
sns.stripplot(data=dual_dist, y='accuracy count', hue='Dataset', x='Map loss', dodge=True, 
              size=3, edgecolor='black', linewidth=0.5, alpha=0.4, legend=False)
plt.plot([-0.5, 1.5], [20, 20], '--', color='gray', label='Chance')
plt.title('Dual Stream on Ignore Distractors')
plt.ylabel('Accuracy (%)')
plt.legend()
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
plt.savefig('dualstream_dist_map.png', dpi=300, bbox_inches='tight')

#%% *************** Figure S1. The effect of the map objective *****************
fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(11, 9))
ghibli.set_palette('PonyoMedium')
sns.barplot(data=dual_simp, y='accuracy count', hue='Dataset', x='Map loss', ax=axs[0,0], legend=False)
ghibli.set_palette('PonyoLight')
sns.stripplot(data=dual_simp, y='accuracy count', hue='Dataset', x='Map loss', dodge=True,
              size=5, edgecolor='black', linewidth=1, alpha=0.5, legend=False, ax=axs[0,0])
axs[0,0].plot([-0.5, 1.5], [20, 20], '--', color='gray', label='Chance')
axs[0,0].set_ylabel('Dual-Stream \n Accuracy (%)')
axs[0,0].set_title('Simple Counting')

ghibli.set_palette('PonyoMedium')
sns.barplot(data=dual_dist, y='accuracy count', hue='Dataset', x='Map loss', ax=axs[0,1])
ghibli.set_palette('PonyoLight')
sns.stripplot(data=dual_dist, y='accuracy count', hue='Dataset', x='Map loss', dodge=True,
              size=5, edgecolor='black', linewidth=1, alpha=0.5, legend=False, ax=axs[0,1])
axs[0,1].plot([-0.5, 1.5], [20, 20], '--', color='gray', label='Chance')
axs[0,1].legend(fontsize=15)
axs[0,1].set_title('Ignore Distractors')

ghibli.set_palette('PonyoMedium')
sns.barplot(data=cnn_simp, y='accuracy count', hue='Dataset', x='Map loss', legend=False, ax=axs[1,0])
# sns.swarmplot(data=cnn_nomap, y='accuracy count', x='Task', hue='Dataset', dodge=True,
            #   size=2, edgecolor='black', linewidth=0.5, alpha=0.5, legend=False)
ghibli.set_palette('PonyoLight')
# ghibli.set_palette('PonyoMedium')
# ghibli.set_palette('PonyoDark')
sns.stripplot(data=cnn_simp, y='accuracy count', hue='Dataset', x='Map loss', dodge=True,
              size=5, edgecolor='black', linewidth=1, alpha=0.5, legend=False, ax=axs[1,0])
axs[1,0].plot([-0.5, 1.5], [20, 20], '--', color='gray', label='Chance')
axs[1,0].set_ylabel('CNN \n Accuracy (%)')

ghibli.set_palette('PonyoMedium')
sns.barplot(data=cnn_dist, y='accuracy count', hue='Dataset', x='Map loss', legend=False, ax=axs[1,1])
# sns.swarmplot(data=cnn_nomap, y='accuracy count', x='Task', hue='Dataset', dodge=True,
            #   size=2, edgecolor='black', linewidth=0.5, alpha=0.5, legend=False)
ghibli.set_palette('PonyoLight')
# ghibli.set_palette('PonyoMedium')
# ghibli.set_palette('PonyoDark')
sns.stripplot(data=cnn_dist, y='accuracy count', hue='Dataset', x='Map loss', dodge=True,
              size=5, edgecolor='black', linewidth=1, alpha=0.5, legend=False, ax=axs[1,1])
axs[1,1].plot([-0.5, 1.5], [20, 20], '--', color='gray', label='Chance')

plt.tight_layout()
plt.savefig('map_loss.png', dpi=200, bbox_inches='tight')


#%%
cnn_simp_map.query('epoch==300').groupby('Dataset').mean()
cnn_simp_map.query('epoch==300').groupby('Dataset').std()
cnn_dist_map.query('epoch==300').groupby('Dataset').mean()
cnn_dist_map.query('epoch==300').groupby('Dataset').std()
# %% *********************************** FIGURE 3: Ablations and Controls ******
# Simple Counting
# models to compare: ablatev, ablated, nomap, cnn+map
model_desc_temp = '{}_hsize-1024_input-{}'
data_desc_temp = '{}_num1-5_nl-0.74_policy-{}_trainshapes-{}same_{}_{}_{}_100000'
train_desc_temp = 'loss-{}_opt-Adam_drop0.5_sort_count-{}_300eps'
# Filter data we already loaded to only include OOD both, not the other OOD sets
cnn_simp_oodboth = cnn_simp_nomap.drop(cnn_simp_nomap.query("Dataset == 'OOD Shape' or Dataset == 'OOD Luminance'").index)
dual_simp_oodboth = dual_simp_map.drop(dual_simp_map.query("Dataset == 'OOD Shape' or Dataset == 'OOD Luminance'").index)
cnn_simp_map_oodboth = cnn_simp_map.drop(cnn_simp_map.query("Dataset == 'OOD Shape' or Dataset == 'OOD Luminance'").index)
dual_simp_nomap = dual_simp_nomap.drop(dual_simp_nomap.query("`accuracy count` < 21").index)
dual_simp_oodboth_nomap = dual_simp_nomap.drop(dual_simp_nomap.query("Dataset == 'OOD Shape' or Dataset == 'OOD Luminance'").index)



model_desc = model_desc_temp.format('rnn_classifier2stream', 'xy')
data_desc = data_desc_temp.format('logpolar', 'cheat+jitter', 'BCDE', '', 'logpolar', 12)
train_desc = train_desc_temp.format('both', 'multi')
ablatev_simp_map = plot_utils.get_results(nreps, model_desc, data_desc, train_desc)
train_desc = train_desc_temp.format('both', 'notA')
ablatev_simp_map2 = plot_utils.get_results(nreps, model_desc, data_desc, train_desc)
if isinstance(ablatev_simp_map2, pd.DataFrame):
    ablatev_simp_map = pd.concat((ablatev_simp_map, ablatev_simp_map2))
ablatev_simp_map = ablatev_simp_map.query('epoch==300')
ablatev_simp_map['Map loss'] = True
ablatev_simp_map['Task'] = 'Simple \nCounting'
ablatev_simp_map['Dataset'] = dual_simp_map.apply(lambda x: plot_utils.relabel_dataset(x.dataset, x['test shapes'], 
                                                                                  x['test lums']), axis=1)
ablatev_simp_map['Model'] = 'Ablate contents'

train_desc = train_desc_temp.format('both', 'multi')
model_desc = model_desc_temp.format('rnn_classifier2stream', 'shape')
ablated_simp_map = plot_utils.get_results(nreps, model_desc, data_desc, train_desc)
ablated_simp_map = ablated_simp_map.query('epoch==300')
ablated_simp_map['Map loss'] = True
ablated_simp_map['Task'] = 'Simple \nCounting'
ablated_simp_map['Dataset'] = dual_simp_map.apply(lambda x: plot_utils.relabel_dataset(x.dataset, x['test shapes'], 
                                                                                  x['test lums']), axis=1)
ablated_simp_map['Model'] = 'Ablate position'

model_desc = model_desc_temp.format('recurrent_control', 'shape')
data_desc = data_desc_temp.format('noise', 'cheat+jitter', 'BCDE', '', 'gw6', 12)
train_desc = train_desc_temp.format('both', 'notA')
# recurrent_control_hsize-1024_input-shape_noise_num1-5_nl-0.74_policy-cheat+jitter_trainshapes-BCDEsame__gw6_12_100000_loss-both_opt-Adam_drop0.5_count-notA_300eps_rep19_ep-300
recurrent_ctrl = plot_utils.get_results(nreps, model_desc, data_desc, train_desc)
recurrent_ctrl = recurrent_ctrl.query('epoch==300')
recurrent_ctrl['Map loss'] = True
recurrent_ctrl['Task'] = 'Simple \nCounting'
recurrent_ctrl['Dataset'] = recurrent_ctrl.apply(lambda x: plot_utils.relabel_dataset(x.dataset, x['test shapes'], 
                                                                                 x['test lums']), axis=1)
recurrent_ctrl['Model'] = 'Whole image RNN'

model_desc = model_desc_temp.format('mlp', 'shape')
mlp = plot_utils.get_results(nreps, model_desc, data_desc, train_desc)
mlp = mlp.query('epoch==300')
mlp['Map loss'] = True
mlp['Task'] = 'Simple \nCounting'
mlp['Dataset'] = mlp.apply(lambda x: plot_utils.relabel_dataset(x.dataset, x['test shapes'], x['test lums']), axis=1)
mlp['Model'] = 'MLP+map'

data_desc_temp_misalign = '{}_num1-5_nl-0.74_policy-{}_trainshapes-{}same_{}_{}_{}_misalign_100000'
model_desc = model_desc_temp.format('rnn_classifier2stream', 'both')
data_desc = data_desc_temp_misalign.format('logpolar', 'cheat+jitter', 'BCDE', '', 'logpolar', 12)
train_desc = train_desc_temp.format('both', 'notA')
misalign_simp = plot_utils.get_results(nreps, model_desc, data_desc, train_desc)
misalign_simp = misalign_simp.query('epoch==300')
misalign_simp['Map loss'] = True
misalign_simp['Task'] = 'Simple \nCounting'
misalign_simp['Dataset'] = misalign_simp.apply(lambda x: plot_utils.relabel_dataset(x.dataset, x['test shapes'], x['test lums']), axis=1)
misalign_simp['Model'] = 'Streams misaligned'

simple = pd.concat((cnn_simp_oodboth, dual_simp_oodboth, ablatev_simp_map,
                    ablated_simp_map, cnn_simp_map_oodboth, dual_simp_oodboth_nomap,
                    recurrent_ctrl, misalign_simp))
simple.groupby(['Model', 'Dataset']).agg(Mean=('accuracy count', np.mean),
                                           Std=('accuracy count', np.std))
simple = simple.query("Dataset == 'OOD Both' or Dataset == 'Validation' or Dataset == 'Train'")
simple = simple.reset_index()

# simple

#%% plot
# sns.set_palette(sns.color_palette(colors2))
ghibli.set_palette('PonyoMedium2')
sns.barplot(data=simple, y='Model', x='accuracy count', hue='Dataset', dodge=False)
plt.xlabel('Accuracy (%)')
# plt.plot([20, 20], [-0.5, 5.5], '--', color='gray', label='chance')
plt.plot([20, 20], [-0.5, 7.5], '--', color='gray', label='Chance')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
plt.title('Simple Counting')
plt.savefig('simple_ablations.png', dpi=300, bbox_inches='tight')

# %% Ignore Distractors

# filter
# cnn_dist_oodboth = cnn_dist_nomap.drop(cnn_dist_nomap.query("Dataset == 'OOD Shape' or Dataset == 'OOD Luminance'").index)
# dual_dist_oodboth = dual_dist_map.drop(dual_dist_map.query("Dataset == 'OOD Shape' or Dataset == 'OOD Luminance'").index)
# cnn_dist_map_oodboth = cnn_dist_map.drop(cnn_dist_map.query("Dataset == 'OOD Shape' or Dataset == 'OOD Luminance'").index)
# dual_dist_oodboth_nomap = dual_dist_nomap.drop(dual_dist_nomap.query("Dataset == 'OOD Shape' or Dataset == 'OOD Luminance'").index)

model_desc = model_desc_temp.format('pretrained_ventral-cnn-mse', 'xy')
data_desc = data_desc_temp.format('logpolar', 'cheat+jitter', 'BCDE', 'distract012', 'logpolar', 12)
train_desc = train_desc_temp.format('both', 'multi')
ablatev_dist_map = plot_utils.get_results(nreps, model_desc, data_desc, train_desc)
train_desc = train_desc_temp.format('both', 'notA') # I should really just rename the files with 'multi' to say 'notA' but this is easier
ablatev_dist_map2 = plot_utils.get_results(nreps, model_desc, data_desc, train_desc)
if isinstance(ablatev_dist_map2, pd.DataFrame):
    ablatev_dist_map = pd.concat((ablatev_dist_map, ablatev_dist_map2))
ablatev_dist_map = ablatev_dist_map.query('epoch==300')
ablatev_dist_map['Map loss'] = True
ablatev_dist_map['Task'] = 'Ignore \nDistractors'
ablatev_dist_map['Dataset'] = ablatev_dist_map.apply(lambda x: plot_utils.relabel_dataset(x.dataset, x['test shapes'], x['test lums']), axis=1)
ablatev_dist_map['Model'] = 'Ablate contents'
# ablatev_dist_map = ablatev_dist_map.drop(ablatev_dist_map.query("Dataset == 'OOD Shape' or Dataset == 'OOD Luminance'").index)
train_desc = train_desc_temp.format('both', 'multi')
model_desc = model_desc_temp.format('pretrained_ventral-cnn-mse', 'shape')
ablated_dist_map = plot_utils.get_results(nreps, model_desc, data_desc, train_desc)
train_desc = train_desc_temp.format('both', 'notA')
ablated_dist_map2 = plot_utils.get_results(nreps, model_desc, data_desc, train_desc)
if isinstance(ablated_dist_map2, pd.DataFrame):
    ablated_dist_map = pd.concat((ablated_dist_map, ablated_dist_map2))
ablated_dist_map = ablated_dist_map.query('epoch==300')
ablated_dist_map['Map loss'] = True
ablated_dist_map['Task'] = 'Ignore \nDistractors'
ablated_dist_map['Dataset'] = ablated_dist_map.apply(lambda x: plot_utils.relabel_dataset(x.dataset, x['test shapes'], x['test lums']), axis=1)
# ablated_dist_map = ablated_dist_map.drop(ablated_dist_map.query("Dataset == 'OOD Shape' or Dataset == 'OOD Luminance'").index)
ablated_dist_map['Model'] = 'Ablate position'

model_desc = model_desc_temp.format('pretrained_ventral-cnn-mse-finetune-nopretrain', 'both')
train_desc = train_desc_temp.format('both', 'notA')
no_pretraining = plot_utils.get_results(nreps, model_desc, data_desc, train_desc)
no_pretraining = no_pretraining.query('epoch==300')
no_pretraining['Map loss'] = True
no_pretraining['Task'] = 'Ignore \nDistractors'
no_pretraining['Dataset'] = no_pretraining.apply(lambda x: plot_utils.relabel_dataset(x.dataset, x['test shapes'], x['test lums']), axis=1)
# no_pretraining = no_pretraining.drop(no_pretraining.query("Dataset == 'OOD Shape' or Dataset == 'OOD Luminance'").index)
no_pretraining['Model'] = 'No ventral pretraining'

data_desc_temp_misalign = '{}_num1-5_nl-0.74_policy-{}_trainshapes-{}same_{}_{}_{}_misalign_100000'
data_desc = data_desc_temp_misalign.format('logpolar', 'cheat+jitter', 'BCDE', 'distract012', 'logpolar', 12)
model_desc = model_desc_temp.format('pretrained_ventral-cnn-mse', 'both')
train_desc = train_desc_temp.format('both', 'notA')
misalign_dist = plot_utils.get_results(nreps, model_desc, data_desc, train_desc)
misalign_dist = misalign_dist.query('epoch==300')
misalign_dist['Map loss'] = True
misalign_dist['Task'] = 'Ignore \nDistractors'
misalign_dist['Dataset'] = misalign_dist.apply(lambda x: plot_utils.relabel_dataset(x.dataset, x['test shapes'], x['test lums']), axis=1)
# no_pretraining = no_pretraining.drop(no_pretraining.query("Dataset == 'OOD Shape' or Dataset == 'OOD Luminance'").index)
misalign_dist['Model'] = 'Streams misaligned'


dist = pd.concat((cnn_dist, dual_dist_map, ablatev_dist_map, 
                    ablated_dist_map, cnn_dist_map, dual_dist_nomap, 
                    no_pretraining, misalign_dist))
dist.groupby(['Model', 'Dataset']).agg(Mean=('accuracy count', np.mean),
                                           Std=('accuracy count', np.std))
dist = dist.reset_index()
dist = dist.drop(dist.query("Dataset == 'OOD Shape' or Dataset == 'OOD Luminance'").index)

# %% plot
# sns.set_palette(sns.color_palette(colors2))
ghibli.set_palette('PonyoMedium2')
sns.barplot(data=dist, y='Model', x='accuracy count', hue='Dataset', dodge=False)
plt.xlabel('Accuracy (%)')
# plt.plot([20, 20], [-0.5, 5.5], '--', color='gray', label='chance')
plt.plot([20, 20], [-0.5, 6.5], '--', color='gray', label='Chance')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
plt.title('Ignore Distractors')
plt.savefig('dist_ablations.png', dpi=300, bbox_inches='tight')

#%% Make more clear plot, omit Train, just plot validation and OODBoth but in separate panels
order = ['Dual-stream RNN', 'Ablate contents', 'Ablate position',
         'Dual-stream no map', 'CNN', 'CNN+map', 'Whole image RNN', 'Streams misaligned']
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(11, 9), sharex=True)
sns.barplot(data=simple.query("Dataset == 'Validation'"), y='Model',
            x='accuracy count', dodge=False, color='#5A6F80', ax=ax1,
            order=order)
sns.stripplot(data=simple.query("Dataset == 'Validation'"), y='Model',
              x='accuracy count', dodge=False, color='#ADB7C0', ax=ax1,
              linewidth=-.7, alpha=0.5, edgecolor='black', order=order)
ax1.plot([20, 20], [-0.5, 7.5], '--', color='gray', label='Chance')
ax1.set_title('Validation')
ax1.set_xlabel('Accuracy')
ax1.set_ylabel('Simple Counting \n\n ')
sns.barplot(data=simple.query("Dataset == 'OOD Both'"), y='Model',
            x='accuracy count', dodge=False, color='#DE7862', ax=ax2,
            order=order)
sns.stripplot(data=simple.query("Dataset == 'OOD Both'"), y='Model',
              x='accuracy count', dodge=False, color='#EEBCB1', ax=ax2,
              linewidth=-.7, alpha=0.5, edgecolor='black', order=order)
ax2.plot([20, 20], [-0.5, 7.5], '--', color='gray', label='Chance')
ax2.set_title("OOD Both")
ax2.set_xlabel('Accuracy')
ax2.set_yticklabels('')
ax2.set_ylabel('')
plt.tight_layout()
dorder = ['Dual-stream RNN', 'Ablate contents', 'Ablate position',
          'Dual-stream no map', 'CNN', 'CNN+map', 'No ventral pretraining',
          'Streams misaligned']
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
sns.barplot(data=dist[dist['Dataset']=='Validation'], y='Model',
            x='accuracy count', dodge=False, color='#5A6F80', ax=ax3,
            order=dorder)
sns.stripplot(data=dist[dist['Dataset']=='Validation'], y='Model',
              x='accuracy count', dodge=False, color='#ADB7C0', ax=ax3,
              linewidth=-.7, alpha=0.5, edgecolor='black', order=dorder)
# ax3.set_title('Validation')
ax3.plot([20, 20], [-0.5, 7.5], '--', color='gray', label='Chance')
ax3.set_xlabel('Accuracy')
ax3.set_ylabel('Ignore Distractors \n ')
sns.barplot(data=dist[dist['Dataset']=='OOD Both'], y='Model',
            x='accuracy count', dodge=False, color='#DE7862', ax=ax4,
            order=dorder)
sns.stripplot(data=dist[dist['Dataset']=='OOD Both'], y='Model',
              x='accuracy count', dodge=False, color='#EEBCB1', ax=ax4,
              linewidth=-.7, alpha=0.5, edgecolor='black', order=dorder)
# ax4.set_title("OOD Both")
ax4.plot([20, 20], [-0.5, 7.5], '--', color='gray', label='Chance')
ax4.set_xlabel('Accuracy')
ax4.set_ylabel('')
ax4.set_yticklabels('')
plt.tight_layout()
plt.savefig('ablations.png', dpi=300, bbox_inches='tight')



# %%
# *********************************************** FIGURE 4 - HUMAN STUFF *************************

# Development
train_desc_temp = 'loss-{}_opt-Adam_drop0.5_sort_count-{}_{}eps'
data_desc_temp = '{}_num1-5_nl-0.74_policy-{}_trainshapes-{}same_{}_{}_{}_100000'
model_desc = model_desc_temp.format('rnn_classifier2stream', 'both')
data_desc = data_desc_temp.format('logpolar', 'cheat+jitter', 'BCDE', '', 'logpolar', 12)
train_desc = train_desc_temp.format('both', 'notA', 300)
dataset = 3
dual_conf_25 = plot_utils.get_confusion(nreps, model_desc, data_desc, train_desc, 25).mean(axis=0)[dataset]
dual_conf_50 = plot_utils.get_confusion(nreps, model_desc, data_desc, train_desc, 50).mean(axis=0)[dataset]
dual_conf_75 = plot_utils.get_confusion(nreps, model_desc, data_desc, train_desc, 75).mean(axis=0)[dataset]

model_desc = model_desc_temp.format('bigcnn', 'shape')
data_desc = data_desc_temp.format('noise', 'cheat+jitter', 'BCDE', '', 'gw6', 12)
# te_data_desc = data_desc_temp.format('both', '5000')
train_desc = train_desc_temp.format('num', 'notA', 300)
cnn_conf_25 = plot_utils.get_confusion(nreps, model_desc, data_desc, train_desc, 25).mean(axis=0)[dataset]
cnn_conf_50 = plot_utils.get_confusion(nreps, model_desc, data_desc, train_desc, 50).mean(axis=0)[dataset]
cnn_conf_75 = plot_utils.get_confusion(nreps, model_desc, data_desc, train_desc, 75).mean(axis=0)[dataset]


fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2,3, sharex=True, sharey=True)
ax1.matshow(dual_conf_25)
ax2.matshow(dual_conf_50)
ax3.matshow(dual_conf_75)
ax4.matshow(cnn_conf_25)
ax5.matshow(cnn_conf_50)
ax6.matshow(cnn_conf_75)
for ax in [ax1, ax2, ax3, ax4, ax5, ax6]: ax.grid(False)
# %%
model_desc = model_desc_temp.format('pretrained_ventral-cnn-mse', 'both')
data_desc = data_desc_temp.format('logpolar', 'cheat+jitter', 'BCDE', 'distract012', 'logpolar', 12)
train_desc_temp = 'loss-{}_opt-Adam_drop0.5_sort_count-{}_{}eps'
train_desc = train_desc_temp.format('both', 'notA', 300)
dataset = 0  # 0 for validation set
dual_conf_25 = plot_utils.get_confusion(nreps, model_desc, data_desc, train_desc, 25).mean(axis=(0, 2))[dataset]
dual_conf_50 = plot_utils.get_confusion(nreps, model_desc, data_desc, train_desc, 50).mean(axis=(0, 2))[dataset]
dual_conf_75 = plot_utils.get_confusion(nreps, model_desc, data_desc, train_desc, 75).mean(axis=(0, 2))[dataset]
dual_conf_end = plot_utils.get_confusion(nreps, model_desc, data_desc, train_desc)[:, -1, :, :, :, :]  # take the last 
maxx = dual_conf_end.max()
dual_conf_end = dual_conf_end.mean(axis=(0, 2))[dataset]

# model_desc = model_desc_temp.format('bigcnn', 'shape')
# data_desc = data_desc_temp.format('noise', 'cheat+jitter', 'BCDE', 'distract012', 'gw6', 12)
# train_desc = train_desc_temp.format('num', 'notA', 10)
# ccn_conf_25 = get_confusion(model_desc, data_desc, train_desc, 26).mean(axis=(0, 2))[dataset]
# ccn_conf_50 = get_confusion(model_desc, data_desc, train_desc, 50).mean(axis=(0, 2))[dataset]
# ccnl_conf_75 = get_confusion(model_desc, data_desc, train_desc, 75).mean(axis=(0, 2))[dataset]
sns.set(font_scale=1.8)
fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, sharex=True, sharey=True, figsize=(10, 26), layout='constrained')
ax1.matshow((dual_conf_25/maxx)*100, vmin=0, vmax=100)
ax1.set_xlabel('Predicted Class')
ax1.set_ylabel('True Class')
ax1.set_title('25% Correct')
ax2.matshow((dual_conf_50/maxx)*100, vmin=0, vmax=100)
ax2.set_title('50% Correct')
ax3.matshow((dual_conf_75/maxx)*100, vmin=0, vmax=100)
ax3.set_title('75% Correct')
pcm = ax4.matshow((dual_conf_end/maxx)*100, vmin=0, vmax=100)
ax4.set_title('Convergence')
for ax in [ax1, ax2, ax3, ax4]:
    ax.grid(False)
    ax.set_xlabel('Predicted \nClass')
    ax.tick_params(top=False, labeltop=False, bottom=False, labelbottom=True)
    ax.set_xticks([0, 1, 2, 3, 4], [1, 2, 3, 4, 5])
    ax.set_yticks([0, 1, 2, 3, 4], [1, 2, 3, 4, 5])
    # plt.tick_params(top='off', bottom='off', left='off', right='off')
# fig.colorbar(pcm, ax=[ax1, ax2, ax3, ax4], shrink=0.8, location='bottom')
fig.colorbar(pcm, ax=ax4, shrink=0.08, location='right', pad=0.1)
# plt.tight_layout()    
plt.savefig('development_confusion.png', dpi=150, bbox_inches='tight')

#%%
# BEHAVIOUR EXP2
sns.set(font_scale=2)
model_desc_temp = '{}_hsize-1024_input-{}'
data_desc_temp = '{}_num3-6_nl-0.74_policy-{}_trainshapes-{}same_{}_{}_{}_100000'
train_desc_temp = 'loss-{}_opt-Adam_drop0.5_sort_count-multi_300eps'
model_desc = model_desc_temp.format('rnn_classifier2stream', 'both')
data_desc = data_desc_temp.format('logpolar_mixed', 'cheat+jitter', 'ESUZFCKJ', '', 'logpolar_mixed', 12)
train_desc = train_desc_temp.format('both')
print('Loading dual stream simple')
hum_simple = plot_utils.get_results(nreps, model_desc, data_desc, train_desc)
hum_simple = hum_simple.query('epoch==300').query("dataset=='validation'")
hum_simple['Task'] = 'Simple \nCounting'

hum_simple_conf = plot_utils.get_confusion(nreps, model_desc, data_desc, train_desc)

model_desc = model_desc_temp.format('pretrained_ventral-cnn-mse', 'both')
data_desc = data_desc_temp.format('logpolar_mixed', 'cheat+jitter', 'ESUZFCKJ', 'distract123', 'logpolar_mixed', 12)
print('Loading dual stream ignore')
hum_ignore = plot_utils.get_results(nreps, model_desc, data_desc, train_desc)
hum_ignore = hum_ignore.query('epoch==300').query("dataset=='validation'")
hum_ignore['Task'] = 'Ignore \nDistractors'

hum_ignore_conf = plot_utils.get_confusion(nreps, model_desc, data_desc, train_desc)

hum = pd.concat((hum_simple, hum_ignore))
hum = hum.rename(columns={"viewing": "View"})
hum = hum.assign(View=hum.View.map({'free': "Free", 'fixed': "Fixed"}))
hum['Viewer'] = 'Dual-Stream RNN \n(with covert saccades)'

# Without covert saccaades
model_desc_temp = '{}_hsize-1024_input-{}'
data_desc_temp = '{}_num3-6_nl-0.74_policy-{}_trainshapes-{}same_{}_{}_{}_100000'
train_desc_temp = 'loss-{}_opt-Adam_drop0.5_sort_count-multi_300eps'
model_desc = model_desc_temp.format('rnn_classifier2stream', 'both')
data_desc = data_desc_temp.format('logpolar_mixed_no_covert', 'cheat+jitter', 'ESUZFCKJ', '', 'logpolar_mixed_no_covert', 12)
train_desc = train_desc_temp.format('both')
print('Loading dual stream simple no covert')
hum_simple_nc = plot_utils.get_results(nreps, model_desc, data_desc, train_desc)
hum_simple_nc = hum_simple_nc.query('epoch==300').query("dataset=='validation'")
hum_simple_nc['Task'] = 'Simple \nCounting'

# hum_simple_conf = get_confusion(model_desc, data_desc, train_desc)

model_desc = model_desc_temp.format('pretrained_ventral-cnn-mse', 'both')
data_desc = data_desc_temp.format('logpolar_mixed_no_covert', 'cheat+jitter', 'ESUZFCKJ', 'distract123', 'logpolar_mixed_no_covert', 12)
print('Loading dual stream ignore no covert')
hum_ignore_nc = plot_utils.get_results(nreps, model_desc, data_desc, train_desc)
hum_ignore_nc = hum_ignore_nc.query('epoch==300').query("dataset=='validation'")
hum_ignore_nc['Task'] = 'Ignore \nDistractors'

# hum_ignore_conf = get_confusion(model_desc, data_desc, train_desc)

hum_nc = pd.concat((hum_simple_nc, hum_ignore_nc,))
# hum_nc = hum_simple_nc
hum_nc = hum_nc.rename(columns={"viewing": "View"})
hum_nc = hum_nc.assign(View=hum_nc.View.map({'free': "Free", 'fixed': "Fixed"}))
hum_nc['Viewer'] = 'Dual-Stream RNN \n(withput covert saccades)'


#  CNN
model_desc = model_desc_temp.format('bigcnn', 'shape')
data_desc = data_desc_temp.format('noise', 'cheat+jitter', 'ESUZFCKJ', '', 'gw6', 12)
train_desc = train_desc_temp.format('num')
print('Loading CNN simple')
hum_cnn_simp = plot_utils.get_results(nreps, model_desc, data_desc, train_desc)
hum_cnn_simp = hum_cnn_simp.query('epoch==300').query("dataset=='validation'").query("viewing=='fixed'")
hum_cnn_simp['Task'] = 'Simple \nCounting'
model_desc = model_desc_temp.format('bigcnn', 'shape')
data_desc = data_desc_temp.format('noise', 'cheat+jitter', 'ESUZFCKJ', 'distract123', 'gw6', 12)
print('Loading CNN ignore')
hum_cnn_ign = plot_utils.get_results(nreps, model_desc, data_desc, train_desc)
hum_cnn_ign = hum_cnn_ign.query('epoch==300').query("dataset=='validation'").query("viewing=='fixed'")
hum_cnn_ign['Task'] = 'Ignore \nDistractors'

hum_cnn = pd.concat((hum_cnn_simp, hum_cnn_ign,))
hum_cnn = hum_cnn.rename(columns={"viewing": "View"})
hum_cnn['Viewer'] = 'CNN'

# Load Human Behaviour
beh_data2 = plot_utils.load_human_beh_exp2()


beh_data2_grouped = beh_data2.groupby(['Subject_Number', 'View', 'Task']).mean('numeric_only').reset_index()
beh_data2_grouped['Viewer'] = 'Human'
#%%
print('Human Anova')
# model = ols("Accuracy ~ Task + View + Task:View", data=beh_data2_grouped).fit()
# anova_results = anova_lm(model, typ=1)
# print(anova_results)
# print('Human Anova')
# model = ols("Accuracy ~ Task + View + Task:View", data=beh_data2_grouped).fit()
# anova_results = anova_lm(model, typ=2)
# print(anova_results)
# print('Human Anova')

model = ols("Accuracy ~ Task + View + Task:View", data=beh_data2_grouped).fit()
anova_results = anova_lm(model, typ=3)
print(anova_results)
beh_data2_grouped['Groups'] = beh_data2_grouped['Task'] + beh_data2_grouped['View']

tukey_twoway = pairwise_tukeyhsd(endog = beh_data2_grouped["Accuracy"],
                                 groups=beh_data2_grouped["Groups"])
# Display the results
tukey_twoway.summary()
# Multiple Comparison of Means - Tukey HSD, FWER=0.05
# group1	group2	meandiff	p-adj	lower	upper	reject
# Ignore DistractorsFixed	Ignore DistractorsFree	0.1657	0.0	0.0886	0.2428	True
# Ignore DistractorsFixed	Simple CountingFixed	0.2031	0.0	0.126	0.2802	True
# Ignore DistractorsFixed	Simple CountingFree	0.2355	0.0	0.1584	0.3126	True
# Ignore DistractorsFree	Simple CountingFixed	0.0374	0.5848	-0.0397	0.1145	False
# Ignore DistractorsFree	Simple CountingFree	0.0698	0.0904	-0.0073	0.1469	False
# Simple CountingFixed	Simple CountingFree	0.0324	0.6904	-0.0447	0.1095	False


print('Dual stream w/ covert Anova')
hum['Accuracy'] = hum['accuracy count']/100
model = ols("Accuracy ~ Task + View + Task:View", data=hum).fit()
anova_results = anova_lm(model, typ=3)
print(anova_results)
hum['Groups'] = hum['Task'] + hum['View']
tukey_twoway = pairwise_tukeyhsd(endog = hum["Accuracy"],
                                 groups=hum["Groups"])
tukey_twoway.summary()
ign_diff = hum_ignore[hum_ignore.viewing == 'free']['accuracy count'].values - hum_ignore[hum_ignore.viewing == 'fixed']['accuracy count'].values
sim_diff = hum_simple[hum_simple.viewing == 'free']['accuracy count'].values - hum_simple[hum_simple.viewing == 'fixed']['accuracy count'].values
mannwhitneyu(ign_diff, sim_diff, alternative='greater')

print('Dual stream w/o covert Anova')
hum_nc['Accuracy'] = hum_nc['accuracy count']/100
model = ols("Accuracy ~ Task + View + Task:View", data=hum_nc).fit()
anova_results = anova_lm(model, typ=3)
print(anova_results)
hum_nc['Groups'] = hum_nc['Task'] + hum_nc['View']
tukey_twoway = pairwise_tukeyhsd(endog = hum_nc["Accuracy"],
                                  groups=hum_nc["Groups"])
tukey_twoway.summary()
ign_diff = hum_ignore_nc[hum_ignore_nc.viewing == 'free']['accuracy count'].values - hum_ignore_nc[hum_ignore_nc.viewing == 'fixed']['accuracy count'].values
sim_diff = hum_simple_nc[hum_simple_nc.viewing == 'free']['accuracy count'].values - hum_simple_nc[hum_simple_nc.viewing == 'fixed']['accuracy count'].values
mannwhitneyu(ign_diff, sim_diff, alternative='greater')

# CNN Anova not really possible because no View variable
# human_exp = pd.concat([hum, beh_data2])

# %% Plot interaction in both human and model
# Hard to overlap the barplot and a swarm plot with catplot, better to just plot them both separately
# sns.set_palette(colorblind)
# ghibli.set_palette('MarnieLight2')
order = ['Simple \nCounting', 'Ignore \nDistractors']
hue_order = ['Free', 'Fixed']
ghibli.set_palette('PonyoMedium3')
sns.set_style("whitegrid")
fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, sharey=True, figsize=(18,5))
g = sns.barplot(data=beh_data2_grouped, y='accuracy count', x='Task', hue='View', ax=ax1,
                order=order, hue_order=hue_order, legend=True)
ghibli.set_palette('PonyoLight3')
sns.stripplot(data=beh_data2_grouped, y='accuracy count', x='Task', hue='View', dodge=True, 
              size=4, edgecolor='black', linewidth=0.7, alpha=0.5, legend=False, ax=ax1,
              order=order, hue_order=hue_order)
# sns.swarmplot(data=beh_data2_grouped, y='accuracy count', x='Task', hue='View', dodge=True, 
#               size=4, edgecolor='black', linewidth=0.5, alpha=0.5, legend=False, ax=ax1,
#               order=order, hue_order=hue_order)
ax1.plot([-0.5,1.5], [25, 25], '--', color='gray')
ax1.set_title('Human')
ax1.set_ylabel('Accuracy (%)')
sns.despine()

# With covert saccades
ghibli.set_palette('PonyoMedium3')
sns.barplot(data=hum, y='accuracy count', x='Task', hue='View', ax=ax2, order=order, 
            hue_order=hue_order, legend=False)
ghibli.set_palette('PonyoLight3')
sns.stripplot(data=hum, y='accuracy count', x='Task', hue='View', dodge=True, 
              size=4, edgecolor='black', linewidth=0.7, alpha=0.5, legend=False, 
              ax=ax2, order=order, hue_order=hue_order)
# sns.swarmplot(data=hum, y='accuracy count', x='Task', hue='View', dodge=True, 
#               size=4, edgecolor='black', linewidth=0.5, alpha=0.5, legend=False, 
#               ax=ax2, order=order, hue_order=hue_order)
ax2.set_ylabel('')
# plt.plot([20, 20], [-0.5, 5.5], '--', color='black', label='chance')
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
ax2.set_title('Dual-Stream RNN\nCovert Attention')
ax2.plot([-0.5,1.5], [25, 25], '--', color='gray')

# Without covert saccades
ghibli.set_palette('PonyoMedium3')
sns.barplot(data=hum_nc, y='accuracy count', x='Task', hue='View', ax=ax3, order=order, 
            hue_order=hue_order, legend=False)
ghibli.set_palette('PonyoLight3')
sns.stripplot(data=hum_nc, y='accuracy count', x='Task', hue='View', dodge=True, 
              size=4, edgecolor='black', linewidth=0.7, alpha=0.5, legend=False, 
              ax=ax3, order=order, hue_order=hue_order)
ax3.set_ylabel('')
# plt.plot([20, 20], [-0.5, 5.5], '--', color='black', label='chance')
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
ax3.set_title('Dual-Stream RNN \n No Covert Attention')
ax3.plot([-0.5,1.5], [25, 25], '--', color='gray')

# CNN
ghibli.set_palette('PonyoLight')
sns.barplot(data=hum_cnn, y='accuracy count', x='Task', ax=ax4, order=order, 
            hue_order=hue_order, legend=False, color='#A6A0A0')

sns.stripplot(data=hum_cnn, y='accuracy count', x='Task', dodge=True, 
              size=4, edgecolor='black', linewidth=0.7, alpha=0.5, legend=False, 
              ax=ax4, order=order, hue_order=hue_order, color='gray')
ax4.set_ylabel('')
# plt.plot([20, 20], [-0.5, 5.5], '--', color='black', label='chance')
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
ax4.set_title('CNN')
ax4.plot([-0.5,1.5], [25, 25], '--', color='gray')


plt.tight_layout()
plt.savefig('Human_beh_exp2.png', dpi=300, bbox_inches='tight')

# %%  EYETRACKING
# Load eye tracking data
# eye1 = load_eye1()

# sns.barplot(data=eye1, y='fraction_unglimpsed', x='correct')
# plt.title('Fraction of items unglimpsed significantly higher on incorrect trials ')
# plt.savefig('unglimpsed-correct.png', dpi=300, bbox_inches='tight')

# eye1['n_fixations'] = eye1['fixation_posX'].apply(len)
# eye1 = eye1[['n_fixations', 'target_numerosity', 'total_numerosity', 'fraction_unglimpsed', 'correct']]
# eye1 = eye1.rename(columns={'target_numerosity':'numerosity_target', 'total_numerosity':'numerosity_total'})
# eye1['Experiment'] = '1'


eye2 = plot_utils.load_eye2()
# eye2 = eye2.assign(Task=eye2.Task.map({'Count All': "Simple Counting", 'Ignore A': "Ignore Distractors"}))
eye2['numerosity_total'] = eye2['numerosity_target'] + eye2['numerosity_dist']
# eye2['numerosity_response'] = eye2.Response
# Drop fixed gaze trials
free = eye2.query("View=='Free'")
free['n_fixations'] = free['fixation_posX_stim'].apply(len)
# free = free[['n_fixations', 'numerosity_target', 'numerosity_total', 'numerosity_response','Task', 'Response']]

# free = free[['n_fixations', 'numerosity_target', 'numerosity_total','Task', 'Response']]
free['Experiment'] = '2'
free['id'] = free.index
width, height = 48,48
free['quantized_posX'] = free.apply(lambda i: [np.round(j*width).astype(int) for j in i['fixation_posX_stim']], axis=1)
free['quantized_posY'] = free.apply(lambda i: [np.round(j*height).astype(int) for j in i['fixation_posY_stim']], axis=1)
free['fixation_map']  = free.apply(lambda x: plot_utils.make_map(x.quantized_posX, x.quantized_posY, width, height), axis=1)

fixed = eye2.query("View=='Fixed'")
fixed['n_fixations'] = fixed['fixation_posX_blank'].apply(len)


# exp12 = pd.concat([eye1, free]).reset_index()
# exp12['id'] = exp12.index

# long = pd.wide_to_long(exp12, stubnames='numerosity', sep='_', i="id", j='Numerosity', suffix=r'\w+')
long = pd.wide_to_long(free, stubnames='numerosity', sep='_', i="id", j='Numerosity', suffix=r'\w+')

# %%
# sns.lineplot(data=eye1, y='n_fixations', x='target_numerosity', label='Exp1 Target')
# sns.lineplot(data=eye1, y='n_fixations', x='total_numerosity', label='Exp1 Total')
# # plt.figure()
# sns.lineplot(data=free, y='n_fixations', x='target_numerosity', label='Exp2 Target')
# sns.lineplot(data=free, y='n_fixations', x='total_numerosity', label='Exp2 Total')


# sns.set_palette('colorblind')
sns.set_palette('tab10')
# ghibli.set_palette('PonyoMedium4')
hue_order = ['total', 'target']
gfg = sns.lineplot(data=long, y='n_fixations', x='numerosity', style='Task', 
                   hue='Numerosity', hue_order=hue_order, errorbar='sd')
gfg.legend(fontsize=15)
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
plt.ylabel('Number of Fixations')
plt.xlabel('Numerosity')
# plt.title()
plt.savefig('number_of_fixations.png', dpi=300, bbox_inches='tight')


# %%

sns.lineplot(data=fixed, y='n_fixations', x='numerosity_target', hue='Subject_Number')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, title='Subject #')
plt.title('Fixations during blank screen after image')

# %%

free_simp = free[free['Task'] =='Simple Counting']
free_ign = free[free['Task'] =='Ignore Distractors']
# Y = free_simp.Response.values.reshape(-1, 1)
# Y = free_simp.numerosity_target.values.reshape(-1, 1)
# X = free_simp.n_fixations.values.reshape(-1, 1)

# Y = free_ign.numerosity_target.values.reshape(-1, 1)
# Y = free_ign.numerosity_total.values.reshape(-1, 1)
# X = free_ign.n_fixations.values.reshape(-1, 1)

Y = free.numerosity_total.values.reshape(-1, 1)
# Y = free.Response.values.reshape(-1, 1)
# Y = free.numerosity_target.values.reshape(-1, 1)
n_fix = free.n_fixations.values.reshape(-1, 1)
X = n_fix
# X = np.stack(free.fixation_map)
# n, _, _ = X.shape
# X = X.reshape(n, -1)
# X = np.concatenate((X, n_fix), axis=1)
# 
regressor = LinearRegression()
regressor.fit(X, Y)
print(regressor.score(X, Y))
# plt.scatter(X, Y, marker="x")
sns.lineplot(data=free, x='n_fixations', y='numerosity_total', errorbar='sd')
plt.plot(X, regressor.predict(X), color='black', label=r'linear fit ($r^2=0.39$)')
plt.xlabel('Number of Fixations')
plt.ylabel('Total Numerosity')
plt.legend()
sns.set_style('white')
sns.despine(offset=10, trim=True)
plt.savefig('n_fixation_regression.png', dpi=300, bbox_inches='tight')

# %%
testset = 'new_both'
# cnn_fifty_file = f'/activations/bigcnn_hsize-1024_input-shape_noise_num1-5_nl-0.74_policy-cheat+jitter_trainshapes-BCDEsame_distract012_gw6_12_100000_loss-num_opt-Adam_drop0.5_sort_count-multi_300eps_rep22_acc50_test-{testset}.npz'
# cnn_seventyfive_file = f'/activations/bigcnn_hsize-1024_input-shape_noise_num1-5_nl-0.74_policy-cheat+jitter_trainshapes-BCDEsame_distract012_gw6_12_100000_loss-num_opt-Adam_drop0.5_sort_count-multi_300eps_rep22_acc75_test-{testset}.npz'
# cnn_end_file = f'/activations/bigcnn_hsize-1024_input-shape_noise_num1-5_nl-0.74_policy-cheat+jitter_trainshapes-BCDEsame_distract012_gw6_12_100000_loss-num_opt-Adam_drop0.5_sort_count-multi_300eps_rep22_trained_test-{testset}.npz'
# cnn_fifty = np.load(home + cnn_fifty_file)
# argmax_pred_num = np.argmax(cnn_fifty['predicted_num'], axis=1)

# print(f'cnn acc: {sum(cnn_fifty["correct"])/len(cnn_fifty["correct"])}')

# confusion = np.zeros((5, 5))
# for pred, true in zip(argmax_pred_num, cnn_fifty['numerosity']):
#     confusion[pred, int(true)] += 1
# plt.matshow(confusion)
# plt.ylabel('Predicted')
# plt.xlabel('True')
# plt.title('CNN at 50')

dual_file = '/pretrained_ventral-cnn-mse_hsize-1024_input-both_logpolar_num1-5_nl-0.74_policy-cheat+jitter_trainshapes-BCDEsame_distract012_logpolar_12_100000_loss-both_opt-Adam_drop0.5_sort_count-multi_300eps_rep0_{}_test-{}.npz'
# dual_75_file = '/pretrained_ventral-cnn-mse_hsize-1024_input-both_logpolar_num1-5_nl-0.74_policy-cheat+jitter_trainshapes-BCDEsame_distract012_logpolar_12_100000_loss-both_opt-Adam_drop0.5_sort_count-multi_300eps_rep0_{}_test-{}.npz'
# dual_end_file = '/pretrained_ventral-cnn-mse_hsize-1024_input-both_logpolar_num1-5_nl-0.74_policy-cheat+jitter_trainshapes-BCDEsame_distract012_logpolar_12_100000_loss-both_opt-Adam_drop0.5_sort_count-multi_300eps_rep0_{}_test-{}.npz'
confusion = np.zeros((3, 5, 5))
for i, milestone in enumerate(['acc50', 'acc75', 'trained']):
    dual = np.load(home + 'activations' + dual_file.format(milestone, testset))
    argmax_pred_num = np.argmax(dual['predicted_num'], axis=1)
    for pred, true in zip(argmax_pred_num, dual['numerosity']):
        confusion[i, pred, int(true)] += 1
    plt.matshow(confusion[i])
    plt.ylabel('Predicted')
    plt.xlabel('True')
    plt.title(f'Dual-stream {milestone}')

# print(f'dual stream acc: {sum(dual50["correct"])/len(dual50["correct"])}')

#%% ************************  MORE DIVERSE STIMULI ************************
# 
model_desc_temp = '{}_hsize-1024_input-{}'
data_desc_temp = '{}_num1-5_nl-0.74_policy-{}_trainshapes-{}same_{}_{}_{}_100000'
train_desc_temp = 'loss-{}_opt-Adam_drop0.5_sort_count-{}_300eps'

# Load Dual stream
model_desc = model_desc_temp.format('rnn_classifier2stream', 'both')
data_desc = data_desc_temp.format('logpolar', 'cheat+jitter', 'BCDE', '', 'logpolar', '-rotated12')
train_desc = train_desc_temp.format('both', 'notA')
model_desc + data_desc + train_desc
dual_simp_map = plot_utils.get_results(nreps, model_desc, data_desc, train_desc)
dual_simp_map = dual_simp_map.query('epoch==300')
dual_simp_map['Map loss'] = True
dual_simp_map['Task'] = 'Simple \nCounting'
dual_simp_map['Dataset'] = dual_simp_map.apply(lambda x: plot_utils.relabel_dataset(x.dataset, x['test shapes'], x['test lums']), axis=1)
dual_simp_map['Model'] = 'Dual-stream RNN'

# Load CNN
model_desc = model_desc_temp.format('bigcnn', 'both')
data_desc = data_desc_temp.format('logpolar', 'cheat+jitter', 'BCDE', '', 'logpolar', '-rotated12')
train_desc = train_desc_temp.format('num', 'notA')
cnn_simp_nomap = plot_utils.get_results(nreps, model_desc, data_desc, train_desc)
cnn_simp_nomap = cnn_simp_nomap.query('epoch==300')
cnn_simp_nomap['Map loss'] = False
cnn_simp_nomap['Task'] = 'Simple \nCounting'
cnn_simp_nomap['Model'] = 'CNN'
cnn_simp_nomap['Dataset'] = cnn_simp_nomap.apply(lambda x: plot_utils.relabel_dataset(x.dataset, x['test shapes'],
                                                                                     x['test lums']), axis=1)

simp_revision = pd.concat([dual_simp_map, cnn_simp_nomap])
#%%
ghibli.set_palette('PonyoMedium')
sns.barplot(data=simp_revision, y='accuracy count', x='Model', hue='Dataset', legend=True)
# sns.swarmplot(data=cnn_nomap, y='accuracy count', x='Task', hue='Dataset', dodge=True, 
            #   size=2, edgecolor='black', linewidth=0.5, alpha=0.5, legend=False)
ghibli.set_palette('PonyoLight')
# ghibli.set_palette('PonyoMedium')
# ghibli.set_palette('PonyoDark')
sns.stripplot(data=simp_revision, y='accuracy count', x='Model', hue='Dataset', dodge=True, 
              size=5, edgecolor='black', linewidth=1, alpha=0.5, legend=False)
plt.plot([-0.5, 1.5], [20, 20], '--', color='gray', label='Chance')
plt.legend(fontsize=15, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
# sns.boxplot(data=cnn_nomap, y='accuracy count', x='task', hue='dataset')
# sns.stripplot(data=cnn_nomap, y='accuracy count', x='task', hue='dataset')
plt.title('More Diverse Stimuli')
plt.ylabel('Simple Counting Accuracy (%)', )
plt.grid('on')
plt.savefig('diverse_stimuli.png', dpi=300, bbox_inches='tight')
simp_revision.groupby(['Model', 'Dataset']).mean()

# %%
ghibli.set_palette('PonyoMedium')
sns.barplot(data=simp, y='accuracy count', x='Model', hue='Dataset', legend=True)
# sns.swarmplot(data=cnn_nomap, y='accuracy count', x='Task', hue='Dataset', dodge=True, 
            #   size=2, edgecolor='black', linewidth=0.5, alpha=0.5, legend=False)
ghibli.set_palette('PonyoLight')
# ghibli.set_palette('PonyoMedium')
# ghibli.set_palette('PonyoDark')
sns.stripplot(data=simp, y='accuracy count', x='Model', hue='Dataset', dodge=True, 
              size=5, edgecolor='black', linewidth=1, alpha=0.5, legend=False)
plt.plot([-0.5, 1.5], [20, 20], '--', color='gray', label='Chance')
plt.legend(fontsize=15, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
# sns.boxplot(data=cnn_nomap, y='accuracy count', x='task', hue='dataset')
# sns.stripplot(data=cnn_nomap, y='accuracy count', x='task', hue='dataset')
plt.title('Original Stimuli')
plt.ylabel('Simple Counting Accuracy (%)', )
plt.grid('on')
plt.savefig('original_stimuli.png', dpi=300, bbox_inches='tight')
# %%
