import torch
import torch.nn.functional as F

def topk_entropy(logits, k=100):
    """
    计算每个位置上基于top-k概率的entropy
    
    Args:
        logits: [L, V]
        k: 选取的top-k数量
        
    Returns:
        entropy: [L]
    """
    L, V = logits.shape
    
    # 1. 获取top-k的值和索引
    if k==-1:
        k=logits.shape[-1]
    topk_values, topk_indices = torch.topk(logits, k=k, dim=-1)  # [L, k]
    
    # 2. 只对top-k的logits进行softmax（更高效）
    topk_probs = F.softmax(topk_values, dim=-1)  # [L, k]
    
    # 3. 计算entropy（只考虑top-k部分）
    entropy = -torch.sum(topk_probs * torch.log(topk_probs + 1e-10), dim=-1)  # [L]
    
    return entropy

def top12_prob_diff(logits, labels):
    """
    计算每个L位置上top1和top2概率的差值
    label 要先shift一下（和一般训练时的shift是反过来的）
    Args:
        logits: [L, V]
        
    Returns:
        diff: [L], top1_prob - top2_prob
        top1_probs: [L], top1的概率值
        top2_probs: [L], top2的概率值
        yprob: [L]，label处的概率值
    """
    L, V = logits.shape
    shifted_labels = torch.cat([labels[1:], torch.tensor([V-1], device=labels.device)])
    # shifted_labels = labels
    # shifted_labels = torch.cat([torch.tensor([V], device=labels.device), labels[:-1]])
    # 计算概率分布
    probs = F.softmax(logits, dim=-1)  # [L, V]
    
    # 获取top-2的概率值和索引
    top2_probs, top2_indices = torch.topk(probs, k=2, dim=-1)  # [L, 2]
    
    # 分离top1和top2的概率
    top1_probs = top2_probs[:, 0]  # [L]
    top2_probs = top2_probs[:, 1]  # [L]
    
    # 计算差值
    diff = top1_probs - top2_probs  # [L]

    labels_expanded = shifted_labels.unsqueeze(-1)
    label_probs = probs.gather(dim=-1, index=labels_expanded)  # [L, 1]
    yprob = label_probs.squeeze(-1)  # [L]

    return diff, top1_probs, top2_probs, yprob

def get_label_rank_practical(logits, labels, max_rank_to_compute=1000):
    """
    实用版本：计算前max_rank_to_compute的精确排名，超过的标记为max_rank_to_compute+1
    """
    L, V = logits.shape
    shifted_labels = torch.cat([labels[1:], torch.tensor([V-1], device=labels.device)])
    # shifted_labels = labels
    # shifted_labels = torch.cat([torch.tensor([151643], device=labels.device), labels[:-1]])
    
    probs = F.softmax(logits, dim=-1)
    label_probs = probs[torch.arange(L), shifted_labels]
    
    # 获取top-k
    k = min(max_rank_to_compute, V)
    topk_values, topk_indices = torch.topk(probs, k=k, dim=-1)
    
    # 初始化所有rank为k+1（表示在top-k之外）
    ranks = torch.full((L,), k + 1, dtype=torch.long, device=logits.device)
    
    # 批量检查每个样本的label是否在top-k中
    for i in range(L):
        matches = (topk_indices[i] == shifted_labels[i]).nonzero(as_tuple=True)
        if len(matches[0]) > 0:
            ranks[i] = matches[0][0] + 1  # 转换为从1开始的排名
    
    return ranks