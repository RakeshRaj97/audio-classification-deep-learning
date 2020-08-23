# engine.py
import torch
import torch.nn as nn
from tqdm import tqdm
from utility import utility, AverageMeter

class Engine:

    @staticmethod
    def train_fn(train_loader, model, optimizer, device, epoch):
        total_loss = AverageMeter()
        accuracies = AverageMeter()

        model.train()

        t = tqdm(train_loader)
        for step, d in enumerate(t):
            spect = d["spect"].to(device)
            targets = d["target"].to(device)

            outputs = model(spect)

            loss = utility.loss_fn(outputs, targets)

            optimizer.zero_grad()

            loss.backward()
            # xm.optimizer_step(optimizer, barrier=True)
            optimizer.step()

            acc, n_position = utility.get_position_accuracy(outputs, targets)

            total_loss.update(loss.item(), n_position)
            accuracies.update(acc, n_position)

            t.set_description(f"Train E:{epoch + 1} - Loss:{total_loss.avg:0.4f} - Acc:{accuracies.avg:0.4f}")

        return total_loss.avg

    @staticmethod
    def valid_fn(valid_loader, model, device, epoch):
        total_loss = AverageMeter()
        accuracies = AverageMeter()

        model.eval()

        t = tqdm(valid_loader)
        for step, d in enumerate(t):
            with torch.no_grad():
                spect = d["spect"].to(device)
                targets = d["target"].to(device)

                outputs = model(spect)

                loss = utility.loss_fn(outputs, targets)

                acc, n_position = utility.get_position_accuracy(outputs, targets)

                total_loss.update(loss.item(), n_position)
                accuracies.update(acc, n_position)

                t.set_description(f"Eval E:{epoch + 1} - Loss:{total_loss.avg:0.4f} - Acc:{accuracies.avg:0.4f}")

        return total_loss.avg, accuracies.avg