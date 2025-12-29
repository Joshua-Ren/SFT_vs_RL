import torch
import numpy as np
from matplotlib import pyplot as plt
import torch.nn.functional as F
from utils.utils_analyze import topk_entropy, top12_prob_diff,get_label_rank_practical
from tqdm import tqdm
import seaborn as sns
from utils.config_read import *
sns.set_theme()

config = load_config("./configs/train_basic.yaml")

MODEL = config['model']['name'].split('/')[1]
SCRATCH_PATH = f'/scratch/joshua52/sft_rl_temp/{MODEL}/'
import os
if not os.path.exists(SCRATCH_PATH):
    os.makedirs(SCRATCH_PATH)

def run(EXP_NAME):
    SFT_OR_RL = EXP_NAME.split('_')[0]
    raw_data = torch.load(f'{SCRATCH_PATH}{EXP_NAME}_logits.pt')
    loaded_data = raw_data[f'{SFT_OR_RL}_logits']
    loaded_label = raw_data[f'{SFT_OR_RL}_labels']

    N_sample = len(loaded_data)

    # ---------- 1. Entropy of each token
        # ------- Stats of all examples
    entropy_list = []
    diff_tp12_list = []
    top1prob_list = []
    top2prob_list = []
    yprob_list = []
    rank_list = []
    entropy_top5_list=[]
    entropy_top10_list=[]
    entropy_top100_list=[]
    entropy_top1W_list=[]
    #entropy_top10_list = []
    for i in tqdm(range(N_sample)):
        logits = loaded_data[i]
        labels = loaded_label[i]
        probs = F.softmax(logits, dim=-1)  # [L, V]
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)  # [L]
        entropy_list.append(entropy)
        entropy_top5 = topk_entropy(logits, k=5)
        entropy_top10 = topk_entropy(logits, k=10)
        entropy_top100 = topk_entropy(logits, k=100)
        entropy_top1W = topk_entropy(logits, k=10000)
        entropy_top5_list.append(entropy_top5)
        entropy_top10_list.append(entropy_top10)
        entropy_top100_list.append(entropy_top100)
        entropy_top1W_list.append(entropy_top1W)
        # import pdb
        # breakpoint()

        diff, top1_prob, top2_prob, yprob = top12_prob_diff(logits, labels)
        diff_tp12_list.append(diff)
        top1prob_list.append(top1_prob)
        top2prob_list.append(top2_prob)
        yprob_list.append(yprob)

        rank = get_label_rank_practical(logits, labels, max_rank_to_compute=1000)
        rank_list.append(rank)

    entropy_all = torch.cat(entropy_list,dim=0)
    entropy_top5_all = torch.cat(entropy_top5_list,dim=0)
    entropy_top10_all = torch.cat(entropy_top10_list,dim=0)    
    entropy_top100_all = torch.cat(entropy_top100_list,dim=0)
    entropy_top1W_all = torch.cat(entropy_top1W_list,dim=0)
    diff_all = torch.cat(diff_tp12_list,dim=0)
    top1prob_all = torch.cat(top1prob_list,dim=0)
    top2prob_all = torch.cat(top2prob_list,dim=0)
    yprob_all = torch.cat(yprob_list,dim=0)
    rank_all = torch.cat(rank_list,dim=0)#

    if not os.path.exists(f'./figs/{MODEL}/{EXP_NAME}'):
        os.makedirs(f'./figs/{MODEL}/{EXP_NAME}')

    plt.clf()
    fig, ax = plt.subplots(2, 3, figsize=(20, 14))
    ax[0][0].scatter(entropy_all, yprob_all, alpha=0.3)
    ax[0][0].set_xlabel('entropy')
    ax[0][0].set_ylabel('yprob')
    ax[0][0].set_title('H_vs_p')

    ax[0][1].scatter(entropy_all, diff_all, alpha=0.3)
    ax[0][1].set_xlabel('entropy')
    ax[0][1].set_ylabel('diff')
    ax[0][1].set_title('H_vs_diff')

    ax[0][2].scatter(top1prob_all, top2prob_all, alpha=0.3)
    ax[0][2].set_xlabel('top1')
    ax[0][2].set_ylabel('top2')
    ax[0][2].set_title('top1_vs_top2')

    ax[1][0].scatter(entropy_all, top1prob_all, alpha=0.3)
    ax[1][0].set_xlabel('entropy')
    ax[1][0].set_ylabel('top1')
    ax[1][0].set_title('H_vs_top1')

    ax[1][1].scatter(diff_all, yprob_all, alpha=0.3)
    ax[1][1].set_xlabel('diff')
    ax[1][1].set_ylabel('yprob')
    ax[1][1].set_title('diff_vs_yprob')

    ax[1][2].scatter(entropy_all, rank_all, alpha=0.3)
    ax[1][2].set_xlabel('entropy')
    ax[1][2].set_ylabel('rank')
    ax[1][2].set_title('entropy_vs_rank')
    ax[1][2].set_yscale('log')

    plt.show()
    plt.savefig(f'./figs/{MODEL}/{EXP_NAME}/scatter_all.png')


    # -------- Entropy v.s. partial Entropy
    plt.clf()
    fig, ax = plt.subplots(2, 2, figsize=(15, 15))
    ax[0][0].scatter(x=entropy_all, y=entropy_top5_all,alpha=0.3)
    #ax[0][0].set_xlabel("entropy", fontsize=16)
    ax[0][0].set_ylabel("entropy_tp5", fontsize=16)
    ax[0][0].set_title('With Top5')

    ax[0][1].scatter(x=entropy_all, y=entropy_top10_all,alpha=0.3)
    #ax[0][1].set_xlabel("entropy", fontsize=16)
    #ax[0][1].set_ylabel("entropy_tp10", fontsize=16)
    ax[0][1].set_title('With Top10')

    ax[1][0].scatter(x=entropy_all, y=entropy_top100_all,alpha=0.3)
    ax[1][0].set_xlabel("entropy", fontsize=16)
    ax[1][0].set_ylabel("entropy_tp100", fontsize=16)
    ax[1][0].set_title('With Top100')

    ax[1][1].scatter(x=entropy_all, y=entropy_top1W_all,alpha=0.3)
    ax[1][1].set_xlabel("entropy", fontsize=16)
    #ax[1][1].set_ylabel("entropy_tp1W", fontsize=16)
    ax[1][1].set_title('With Top1W')
    plt.show()
    plt.savefig(f'./figs/{MODEL}/{EXP_NAME}/scatter_entropy_entropytopK.png')



    plt.clf()
    g=sns.jointplot(x=entropy_all, y=yprob_all,alpha=0.5, color="#4CB391")
    g.ax_joint.set_xlabel("entropy", fontsize=16)
    g.ax_joint.set_ylabel("yprob", fontsize=16)
    plt.show()
    plt.savefig(f'./figs/{MODEL}/{EXP_NAME}/scatter_entropy_yprob.png')

    plt.clf()
    g=sns.jointplot(x=entropy_all, y=rank_all,alpha=0.5, color="#4CB391")
    g.ax_joint.set_xlabel("entropy", fontsize=16)
    g.ax_joint.set_ylabel("rank", fontsize=16)
    plt.show()
    plt.savefig(f'./figs/{MODEL}/{EXP_NAME}/scatter_entropy_rank.png')

    plt.clf()
    g=sns.jointplot(x=yprob_all, y=rank_all,alpha=0.5, color="#4CB391")
    g.ax_joint.set_xlabel("yprob", fontsize=16)
    g.ax_joint.set_ylabel("rank", fontsize=16)
    g.ax_joint.set_yscale('log')
    plt.show()
    plt.savefig(f'./figs/{MODEL}/{EXP_NAME}/scatter_yprob_rank.png')

    # -------------- 统计一下distribution的峰度相关信息
    plt.clf()
    plt.hist(entropy_all)
    plt.show()
    plt.savefig(f'./figs/{MODEL}/{EXP_NAME}/entropy_{EXP_NAME}.png')

    plt.clf()
    plt.hist(diff_all)
    plt.show()
    plt.savefig(f'./figs/{MODEL}/{EXP_NAME}/diff_top12_{EXP_NAME}.png')

    plt.clf()
    plt.hist(top1prob_all)
    plt.show()
    plt.savefig(f'./figs/{MODEL}/{EXP_NAME}/top1_prob.png')

    plt.clf()
    plt.hist(top2prob_all)
    plt.show()
    plt.savefig(f'./figs/{MODEL}/{EXP_NAME}/top2_prob.png')

    plt.clf()
    plt.hist(yprob_all)
    plt.show()
    plt.savefig(f'./figs/{MODEL}/{EXP_NAME}/yprob_{EXP_NAME}.png')

# # -------------- 比一下SFT的全量entropy和Top10 entropy差距大不大
# plt.clf()
# combined_data = np.concatenate([entropy_all, entropy_top10_all])
# min_val = combined_data.min()
# max_val = combined_data.max()
# # 设置统一的bins
# num_bins = 30  # 可以调整这个值
# bins = np.linspace(min_val, max_val, num_bins + 1)
# # 绘制对齐的histogram
# plt.figure(figsize=(10, 6))
# plt.hist(entropy_all, bins=bins, color='red', alpha=0.5, label='all', density=True)
# plt.hist(entropy_top10_all, bins=bins, color='blue', alpha=0.5, label='top10', density=True)
# plt.xlabel('Entropy')
# plt.ylabel('Density')
# plt.title('Entropy Distribution Comparison')
# plt.legend()
# plt.grid(True, alpha=0.3)
# plt.show()
# plt.savefig(f'./figs/{SFT_OR_RL}/hist_sft.png')


# import pdb
# breakpoint()

run('sft')
run('rl_greedy')
run('rl_tmp1')

