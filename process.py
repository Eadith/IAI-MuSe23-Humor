import torch
from torch.utils.data import Dataset

import time
import os
import numpy as np
import torch.optim as optim
from eval import evaluate

class MyDataset(Dataset):
    def __init__(self, data, labels):

        features = [row for row in data]
        self.features = [torch.tensor(f, dtype=torch.float) for f in features]

        self.labels = labels.tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        x = self.features[index]
        y = self.labels[index]
        return x, y
    
# def custom_collate_fn(data):
#     """
#     Custom collate function to ensure that the meta data are not treated with standard collate, but kept as ndarrays
#     :param data:
#     :return:
#     """
#     tensors = [d[:3] for d in data]
#     np_arrs = [d[3] for d in data]
#     coll_features, coll_feature_lens, coll_labels = default_collate(tensors)    # 看看后面是什么再说
#     np_arrs_coll = np.row_stack(np_arrs)
#     return coll_features, coll_feature_lens, coll_labels, np_arrs_coll


def train(model, train_loader, optimizer, loss_fn, use_gpu=False):

    report_loss, report_size = 0, 0    # 记录当前 batch 的 loss 和 size
    total_loss, total_size = 0, 0      # 记录到目前为止的所有 batch 的 loss 和 size

    model.train()
    if use_gpu:
        model.cuda()

    for batch, batch_data in enumerate(train_loader, 1):
        features, labels= batch_data
        batch_size = len(labels)

        if use_gpu:
            features = features.cuda()
            labels = labels.cuda()

        optimizer.zero_grad()

        # preds,_ = model(features, feature_lens)

        preds, z = model(features)    # 只需要一个参数

        loss = loss_fn(preds.squeeze(-1), labels.squeeze(-1))

        loss.backward()
        optimizer.step()

        report_loss += loss.item() * batch_size
        report_size += batch_size

        total_loss += report_loss
        total_size += report_size
        report_loss, report_size, start_time = 0, 0, time.time()

    train_loss = total_loss / total_size    # 该 epoch 的平均 loss
    return train_loss


def save_model(model, model_folder):
    model_file_name = f'mlp_model.pth'
    model_file = os.path.join(model_folder, model_file_name)
    torch.save(model, model_file)
    return model_file


def train_model(task, model, data_loader, epochs, lr, model_path,  use_gpu, loss_fn, eval_fn,
                eval_metric_str, early_stopping_patience, reduce_lr_patience, regularization=0.0):
    
    train_loader, val_loader = data_loader['test'], data_loader['devel']

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=regularization)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', patience=reduce_lr_patience,
                                                        factor=0.5, min_lr=1e-5, verbose=True)
    
    best_val_loss = float('inf')    # 记录最佳 loss
    best_val_score = -1             # 记录最佳 auc
    best_model_file = ''
    early_stop = 0

    for epoch in range(1, epochs + 1):
        print(f'Training for Epoch {epoch}...')
        train_loss = train(model, train_loader, optimizer, loss_fn, use_gpu)
        val_loss, val_score = evaluate(task, model, val_loader, loss_fn=loss_fn, eval_fn=eval_fn, use_gpu=use_gpu)

        print(f'Epoch:{epoch:>3} / {epochs} | [Train] | Loss: {train_loss:>.4f}')
        print(f'Epoch:{epoch:>3} / {epochs} | [Val] | Loss: {val_loss:>.4f} | [{eval_metric_str}]: {val_score:>7.4f}')
        print('-' * 50)

        if val_score > best_val_score:
            early_stop = 0
            best_val_score = val_score
            best_val_loss = val_loss
            best_model_file = save_model(model, model_path)

        else:
            early_stop += 1
            if early_stop >= early_stopping_patience:
                print(f'Note: target can not be optimized for {early_stopping_patience} consecutive epochs, '
                      f'early stop the training process!')
                print('-' * 50)
                break

        lr_scheduler.step(1 - np.mean(val_score))

    print(
          f'Best [Val {eval_metric_str}]:{best_val_score:>7.4f} | Loss: {best_val_loss:>.4f}')
    
    return best_val_loss, best_val_score, best_model_file
