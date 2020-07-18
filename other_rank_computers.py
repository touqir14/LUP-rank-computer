import torch

def rank_torch(A, args=None):
    A_tensor = torch.from_numpy(A).cuda()    
    return torch.matrix_rank(A_tensor).cpu().item()