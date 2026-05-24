import ipdb
import torch
from torch.nn.utils.rnn import unpad_sequence


def unpad_y(y_pred, y_true, lens, task="outcome"):
    raw_device = y_pred.device
    device = torch.device("cpu")
    y_pred, y_true, lens = y_pred.to(device), y_true.to(device), lens.to(device)
    y_pred_unpad = unpad_sequence(y_pred, batch_first=True, lengths=lens)
    # y_pred_stack = torch.vstack(y_pred_unpad).squeeze(dim=-1)
    y_pred_stack = torch.vstack([y[-1] for y in y_pred_unpad]).squeeze(dim=-1)
    y_true_unpad = unpad_sequence(y_true, batch_first=True, lengths=lens)
    # y_true_stack = torch.vstack(y_true_unpad).squeeze(dim=-1)
    if task == "readmission":
        #y_true_stack = torch.vstack([y[-1][2] for y in y_true_unpad]).squeeze(dim=-1)
        # 过滤掉序列长度为0或最后一个时间步标签不足3个的样本
        valid_indices = []
        for i in range(len(y_true_unpad)):
            if len(y_true_unpad[i]) > 0 and len(y_true_unpad[i][-1]) > 2:
                valid_indices.append(i)
    
        if len(valid_indices) == 0:
            raise ValueError("No valid samples for readmission task after filtering!")
    
    # 将预测值升维到 (1,) 再拼接
        preds = [y_pred_unpad[i][-1].unsqueeze(0) for i in valid_indices]
        y_pred_stack = torch.cat(preds, dim=0)
    
    # 将标签值升维到 (1,1) 再堆叠，最后 squeeze 掉最后一维（保持与原逻辑一致）
        labels = [y_true_unpad[i][-1][2].unsqueeze(0).unsqueeze(0) for i in valid_indices]
        y_true_stack = torch.vstack(labels).squeeze(dim=-1)
    
        return y_pred_stack, y_true_stack
    elif task == "outcome":
        # Each y step may contain multiple targets; take Outcome as the first column.
        y_true_stack = torch.tensor([y[-1][0].item() for y in y_true_unpad])
    else:
        y_true_stack = torch.vstack([y[-1] for y in y_true_unpad]).squeeze(dim=-1)
    return y_pred_stack.to(raw_device), y_true_stack.to(raw_device)

def unpad_batch(x, y, lens):
    x = x.detach().cpu()
    y = y.detach().cpu()
    lens = lens.detach().cpu()
    x_unpad = unpad_sequence(x, batch_first=True, lengths=lens)
    # x_stack = torch.vstack(x_unpad).squeeze(dim=-1)
    x_stack = torch.vstack([x[-1] for x in x_unpad]).squeeze(dim=-1)
    y_unpad = unpad_sequence(y, batch_first=True, lengths=lens)
    # y_stack = torch.vstack(y_unpad).squeeze(dim=-1)
    y_stack = torch.vstack([y[-1][0] for y in y_unpad]).squeeze(dim=-1)
    return x_stack.numpy(), y_stack.numpy()