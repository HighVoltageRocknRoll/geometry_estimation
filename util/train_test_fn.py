from __future__ import print_function, division
import numpy as np
from tqdm import tqdm
import warnings

def get_lr(optimizer):
    return optimizer.param_groups[0]['lr']

def train(epoch, model, loss_fn, optimizer,
          dataloader, pair_generation_tnf,
          log_interval=50, tb_writer=None, scheduler=False, is_cosine_scheduler=False):
    """
    Main function for training

    :param epoch: int, epoch index
    :param model: pytorch model object
    :param loss_fn: loss function of the model
    :param optimizer: optimizer of the model
    :param dataloader: DataLoader object
    :param pair_generation_tnf: Function to serve couples of samples
    :param log_interval: int, number of steps before logging scalars
    :param tb_writer: pytorch TensorBoard SummaryWriter
    :param scheduler: Eventual Learning rate scheduler

    :return: float, avg value of loss fn over epoch
    """

    model.train()
    train_loss = 0
    for batch_idx, batch in enumerate(tqdm(dataloader, desc='Epoch {}'.format(epoch))):
        optimizer.zero_grad()
        tnf_batch = pair_generation_tnf(batch)
        theta = model(tnf_batch)

        if loss_fn._get_name() == 'MSELoss':
            batch_size = theta.shape[0]
            loss = loss_fn(theta, tnf_batch['theta_GT'].view(batch_size,-1))
        elif loss_fn._get_name() == 'ReconstructionLoss':
            loss = loss_fn(theta, tnf_batch['img_R'], tnf_batch['img_R_orig'])
        elif loss_fn._get_name() == 'CombinedLoss':
            if 'img_R' in tnf_batch and 'img_R_orig' in tnf_batch:
                loss, loss_parts = loss_fn(theta, tnf_batch['theta_GT'], tnf_batch['img_R'], tnf_batch['img_R_orig'])
            else:
                loss, loss_parts = loss_fn(theta, tnf_batch['theta_GT'])
            if tb_writer and batch_idx % log_interval == 0:
                for key, value in loss_parts.items():
                    tb_writer.add_scalar(key, value, (epoch - 1) * len(dataloader) + batch_idx)
        else:
            loss = loss_fn(theta, tnf_batch['theta_GT'])

        loss.backward()
        optimizer.step()

        if is_cosine_scheduler:
            scheduler.step()
        if tb_writer:
            tb_writer.add_scalar('learning rate',
                                 get_lr(optimizer),
                                # scheduler.get_lr()[-1],
                                 (epoch - 1) * len(dataloader) + batch_idx)

        train_loss += loss.data.cpu().numpy().item()

        # log every log_interval
        if batch_idx % log_interval == 0:
            print('\tLoss: {:.6f}'.format(loss.data.item()))
            if tb_writer:
                tb_writer.add_scalar('training loss',
                                     loss.data.item(),
                                     (epoch - 1) * len(dataloader) + batch_idx)

    train_loss /= len(dataloader)
    print('Train set: Average loss: {:.8f}'.format(train_loss))
    return train_loss


def validate_model(model, loss_fn,
                   dataloader, pair_generation_tnf,
                   epoch, tb_writer=None):

    model.eval()
    val_loss = 0
    for batch_idx, batch in enumerate(dataloader):
        tnf_batch = pair_generation_tnf(batch)
        theta = model(tnf_batch)

        if loss_fn._get_name() == 'MSELoss':
            batch_size = theta.shape[0]
            loss = loss_fn(theta, tnf_batch['theta_GT'].view(batch_size,-1))
        elif loss_fn._get_name() == 'ReconstructionLoss':
            loss = loss_fn(theta, tnf_batch['img_R'], tnf_batch['img_R_orig'])
        elif loss_fn._get_name() == 'CombinedLoss':
            if 'img_R' in tnf_batch and 'img_R_orig' in tnf_batch:
                loss, _ = loss_fn(theta, tnf_batch['theta_GT'], tnf_batch['img_R'], tnf_batch['img_R_orig'])
            else:
                loss, _ = loss_fn(theta, tnf_batch['theta_GT'])
        else:
            loss = loss_fn(theta, tnf_batch['theta_GT'])

        val_loss += loss.data.cpu().numpy().item()

    val_loss /= len(dataloader)
    print('Validation set: Average loss: {:.8f}'.format(val_loss))
    if tb_writer:
        tb_writer.add_scalar('val loss',
                             val_loss,
                             epoch)

    return val_loss
