import torch
from torch.utils.tensorboard import SummaryWriter


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].flatten().float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def convert_secs2time(epoch_time):
    need_hour = int(epoch_time / 3600)
    need_mins = int((epoch_time - 3600 * need_hour) / 60)
    need_secs = int(epoch_time - 3600 * need_hour - 60 * need_mins)
    return need_hour, need_mins, need_secs

class Trainer:
    def __init__(self, total_epoch, log_dir='runs'):
        self.total_epoch = total_epoch
        self.current_epoch = 0
        self.writer = SummaryWriter(log_dir)
        self.max_acc = [0.0, 0.0]
        self.max_acc_epoch = [0, 0]

    def update(self, idx, train_loss, train_acc, val_loss, val_acc):
        assert 0 < idx <= self.total_epoch, f"total_epoch: {self.total_epoch}, but update with the {idx} index"
        
        self.writer.add_scalar('Loss/train', train_loss, idx)
        self.writer.add_scalar('Loss/val', val_loss, idx)
        self.writer.add_scalar('Accuracy/train', train_acc, idx)
        self.writer.add_scalar('Accuracy/val', val_acc, idx)
        
        self.current_epoch = idx

        is_best = False
        if self.max_acc[0] < train_acc:
            self.max_acc[0] = train_acc
            self.max_acc_epoch[0] = idx
        if self.max_acc[1] < val_acc:
            self.max_acc[1] = val_acc
            self.max_acc_epoch[1] = idx
            is_best = True

        return is_best

    def max_accuracy(self, is_train):
        if self.current_epoch <= 0:
            return 0.0

        if is_train:
            return self.max_acc[0]
        else:
            return self.max_acc[1]

    def close(self):
        self.writer.close()
