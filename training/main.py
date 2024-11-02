import os
import shutil
import sys
from numpy import record
import torch
from torchvision import datasets, transforms
import argparse
import random

from time import time, strftime

import torch.nn as nn
import torch.optim as optim

from utils import Trainer, AverageMeter, accuracy, convert_secs2time
from models import fc
from losses import SqrHingeLoss


def parse_args(cmd_args):
    parser = argparse.ArgumentParser(description="MNIST Training")

    parser.add_argument(
        "--batch_size",
        type=int,
        default=100,
        help="input batch size for training (default: 100)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.02,
        help="learning rate (default: 0.02)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1000,
        help="number of epochs to train (default: 1000)",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=random.randint(1, 1000000),
        help="Random seed",
    )
    parser.add_argument(
        "--print_freq",
        default=200,
        type=int,
        metavar="N",
        help="Print frequency (default: 200)",
    )
    parser.add_argument(
        "--save_freq",
        default=100,
        type=int,
        metavar="N",
        help="Save frequency (default: 100)",
    )
    parser.add_argument(
        "--data_path", type=str, default="./data/", help="Folder to save datasets"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="./checkpoints/",
        help="Folder to save checkpoints and logs",
    )
    parser.add_argument(
        "--evaluate", action="store_true", help="Evaluate model on validation set"
    )
    parser.add_argument("--loss", default="SqrHinge", type=str, help="Loss function")
    parser.add_argument("--optimizer", default="Adam", type=str, help="Optimizer (Adam or SGD)")
    parser.add_argument("--momentum", default=0.9, type=float, help="Momentum (SGD only, default: 0.9)")
    parser.add_argument("--weight_decay", default=0, type=float, help="Weight decay (default: 0)")
    parser.add_argument(
        "--milestones", type=str, default='[50,100,150,200,250]', help="Scheduler milestones")
    parser.add_argument("--gamma", default=0.25, type=float, help="Scheduler gamma (default: 0.25)")

    args = parser.parse_args(cmd_args)
    args.use_cuda = torch.cuda.is_available()

    args.milestones = eval(args.milestones)

    return args


def train(train_loader, model, criterion, optimizer, epoch, log, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    start_time = time()
    for i, (input, target) in enumerate(train_loader):
        if args.use_cuda:
            input = input.cuda()
            target = target.cuda()

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # compute gradient and do gradient step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if hasattr(model, 'clip_weights'):
            model.clip_weights(-1, 1)

        # measure elapsed time
        batch_time.update(time() - start_time)
        start_time = time()

        if (i + 1) % args.print_freq == 0 or (i+1) == len(train_loader):
            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.detach(), target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            display_log(
                " epoch: [{:03d}][{:03d}/{:03d}] time {batch_time.val:.3f} ({batch_time.avg:.3f}) loss {loss.val:.3f} ({loss.avg:.3f})"
                " prec@1 {top1.val:.3f} ({top1.avg:.3f}) prec@5 {top5.val:.3f} ({top5.avg:.3f})".format(
                    epoch,
                    i + 1,
                    len(train_loader),
                    batch_time=batch_time,
                    loss=losses,
                    top1=top1,
                    top5=top5,
                ),
                log
            )
    display_log(
        "  *** Train *** prec@1 {top1.avg:.3f} prec@5 {top5.avg:.3f} error@1 {error1:.3f}".format(
            top1=top1, top5=top5, error1=100 - top1.avg
        ),
        log
    )

    return top1.avg, losses.avg


def validate(val_loader, model, criterion, log, args):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            if args.use_cuda:
                input = input.cuda()
                target = target.cuda()

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

    display_log(
        "  ** Test ** prec@1 {top1.avg:.3f} prec@5 {top5.avg:.3f} error@1 {error1:.3f}".format(
            top1=top1, top5=top5, error1=100 - top1.avg
        ),
        log,
    )

    return top1.avg, losses.avg


def main():
    args = parse_args(sys.argv[1:])

    start_epoch = 1 #> TODO: add argument

    time_stamp = strftime("%Y%m%d-%H%M%S")

    log = open(
        os.path.join(
            args.save_path,
            "training_log_{0}_{1}.txt".format(args.random_seed, time_stamp),
        ),
        "w",
    )
    display_log("Random Seed: {}".format(args.random_seed), log)
    display_log("Arguments: {}".format(args), log)

    setup_random_seed(args)

    # Compose the transformations
    transform = transforms.Compose(
        [transforms.ToTensor()]  # , transforms.Lambda(lambda x: (x > 0.5).float())
    )

    # Load the MNIST dataset with the custom transformation
    train_dataset = datasets.MNIST(
        root=args.data_path, train=True, transform=transform, download=True
    )
    test_dataset = datasets.MNIST(root=args.data_path, train=False, transform=transform)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=args.batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=args.batch_size, shuffle=False
    )

    if args.loss.lower() == "sqrhinge":
        criterion = SqrHingeLoss()
    elif args.loss.lower() == "crossentropy":
        criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError(f"{args.loss} not supported.")

    if args.use_cuda:
        criterion = criterion.cuda()

    # Create an instance of the model
    model = fc()
    if args.use_cuda:
        model = model.cuda()

    if args.optimizer.lower() == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        raise ValueError(f"{args.optimizer} not supported.")

    if args.evaluate:
        model = fc()
        package = torch.load("model.pth.tar", map_location='cpu') #> TODO: mske this an argument
        model_state_dict = package['state_dict']
        model.load_state_dict(model_state_dict) # , strict=args.strict

        model.eval()
        if args.use_cuda:
            model = model.cuda()

        validate(test_loader, model, criterion, log, args)
        return

    #> normalize? mean=0.1307 and deviation=0.3081.

    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=args.milestones, gamma=args.gamma
    )

    # Train the model
    start_time = time()
    epoch_time = AverageMeter()
    
    trainer = Trainer(total_epoch=args.epochs)

    for epoch in range(start_epoch, args.epochs + 1):
        curr_lr = float(scheduler.get_last_lr()[-1])

        need_hour, need_mins, need_secs = convert_secs2time(
            epoch_time.avg * (args.epochs - epoch)
        )
        need_time = "[Need: {:02d}:{:02d}:{:02d}]".format(
            need_hour, need_mins, need_secs
        )

        display_log(
            "\n[Epoch={:03d}/{:03d}] {:s} [lr={:.6f}] [Error={:.3f}]".format(
                epoch, args.epochs, need_time, curr_lr, 100 - trainer.max_accuracy(False)
            ),
            log,
        )

        # train for one epoch
        train_acc, train_loss = train(
            train_loader, model, criterion, optimizer, epoch, log, args
        )

        scheduler.step()

        # evaluate on validation set
        val_acc, val_loss = validate(test_loader, model, criterion, log, args)
        is_best = trainer.update(epoch, train_loss, train_acc, val_loss, val_acc)

        save_checkpoint(
            {
                "epoch": epoch,
                "random_seed": args.random_seed,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_prec1": trainer.max_accuracy(False)
            },
            is_best,
            args,
            "checkpoint_{0}_{1}_{2}.pth.tar".format(epoch, args.random_seed, time_stamp),
            time_stamp,
        )

        # measure elapsed time
        epoch_time.update(time() - start_time)
        start_time = time()

    trainer.close()
    log.close()


def display_log(print_string, log):
    print("{}".format(print_string))
    log.write("{}\n".format(print_string))
    log.flush()


def save_checkpoint(state, is_best, args, filename, timestamp=""):
    epoch = state["epoch"]
    if epoch % args.save_freq == 0 or args.epochs == epoch:
        filename = os.path.join(args.save_path, filename)
        torch.save(state, filename)

        if is_best:
            bestname = os.path.join(args.save_path, "model_best_{0}.pth.tar".format(timestamp))
            shutil.copyfile(filename, bestname)

def setup_random_seed(args):
    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    if args.use_cuda:
        torch.cuda.manual_seed_all(args.random_seed)


if __name__ == "__main__":
    main()
