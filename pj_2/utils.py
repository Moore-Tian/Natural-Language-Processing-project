import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def batch_iter(data, batch_size=32, shuffle=True):
    data_size = len(data)
    indices = list(range(data_size))
    if shuffle:
        random.shuffle(indices)
    batch_num = (data_size + batch_size - 1) // batch_size
    for i in range(batch_num):
        batch = [data[idx] for idx in indices[i * batch_size: (i + 1) * batch_size]]
        batch = sorted(batch, key=lambda x: len(x[0]), reverse=True)
        sentences = [x[0] for x in batch]
        tags = [x[1] for x in batch]
        yield sentences, tags


def pad(data, padded_token, device):
    lengths = [len(sent) for sent in data]
    max_len = lengths[0]
    padded_data = []
    for s in data:
        padded_data.append(s + [padded_token] * (max_len - len(s)))
    return torch.tensor(padded_data, device=device), lengths


def count_matching(list1, list2):
    count = sum(1 for x, y in zip(list1, list2) if x == y)
    return count


def plot_confusion_matrix(confusion_matrix, labels, plot_name):
    fig, ax = plt.subplots()
    im = ax.imshow(confusion_matrix, cmap='Blues')

    # 添加颜色条
    cbar = ax.figure.colorbar(im, ax=ax)

    # 设置坐标轴标签
    ax.set(xticks=np.arange(confusion_matrix.shape[1]),
           yticks=np.arange(confusion_matrix.shape[0]),
           xticklabels=labels, yticklabels=labels,
           xlabel='Predicted label', ylabel='Actual label')

    # 在格子中显示数值
    thresh = confusion_matrix.max() / 2.
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            ax.text(j, i, format(confusion_matrix[i, j], '.2f'),
                    ha="center", va="center",
                    color="white" if confusion_matrix[i, j] > thresh else "black", fontsize=6)

    # 设置图像标题
    ax.set_title(plot_name)
    # 紧凑显示坐标轴标签
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(plot_name + '.png')


def compute_confusion_matrix(actual_labels, predicted_labels, num_classes):
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
    for actual, predicted in zip(actual_labels, predicted_labels):
        confusion_matrix[actual][predicted] += 1
    row_sums = confusion_matrix.sum(axis=1, keepdims=True)
    normalized_matrix = confusion_matrix / row_sums
    return normalized_matrix


def show_confusion_matrix(true_path, pred_path, num_classes, tag_mapping, plot_name):
    df_true = pd.read_csv(true_path)
    true_labels = df_true['expected'].to_list()
    true_labels = tag_mapping.encode(true_labels)
    df_pred = pd.read_csv(pred_path)
    predicted_labels = df_pred['expected'].to_list()
    predicted_labels = tag_mapping.encode(predicted_labels)
    confusion_matrix = compute_confusion_matrix(true_labels, predicted_labels, num_classes)
    plot_confusion_matrix(confusion_matrix, range(num_classes), plot_name)


# 绘制train_losses和valid_losses曲线图并保存
def plot_losses(train_losses, valid_losses, plot_name):
    plt.plot(train_losses, label='train loss')
    plt.plot(valid_losses, label='valid loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(plot_name + '.png')


# 计算在验证集上的loss
def compute_valid_loss(model, valid_data, batch_size, device, vocab_size, tag_size):
    loss = 0
    num_iter = 0
    for sentences, tags in batch_iter(valid_data, batch_size=batch_size):
        num_iter += 1
        sentences, sent_lengths = pad(sentences, vocab_size - 1, device)
        tags, _ = pad(tags, tag_size - 1, device)

        batch_loss = model(sentences, tags, sent_lengths)  # shape: (b,)
        loss += batch_loss.mean().item()
    return loss / num_iter