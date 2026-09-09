import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import torch
from opacus import PrivacyEngine, GradSampleModule
from opacus.optimizers import DPOptimizer

from loss_traces.config import MODEL_DIR, STORAGE_DIR


class Trainer:
    def __init__(self, args, dataloaders, device):
        self.args = args
        self.trainloader, self.plainloader, self.testloader = dataloaders
        self.num_epochs = args.epochs
        self.device = device
        self.loss_func = torch.nn.CrossEntropyLoss()
        self.loss_func_sample = torch.nn.CrossEntropyLoss(reduction='none')

    @staticmethod
    def get_optimizer(training_params, lr, momentum, weight_decay):
        optimizer = torch.optim.SGD(
            training_params,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay
        )
        return optimizer

    @staticmethod
    def get_training_params(model):
        num_params = 0
        num_training_params = 0
        training_params = []
        for p in model.parameters():
            num_params += p.numel()
            if p.requires_grad:
                training_params.append(p)
                num_training_params += p.numel()
        print(f'Training {num_training_params / (10 ** 6)}M parameters out of {num_params / (10 ** 6)}M')
        return training_params

    def get_all_losses(self, model, args):
        all_losses = []
        all_confs = []
        all_norms = []
        all_csg = []  # Tracks CSG norms

        for inputs, targets, _indices in self.plainloader:
            self.optimizer.zero_grad()
            model.zero_grad()
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # REQUIREMENT FOR CSG: Enable gradient tracking on inputs
            inputs.requires_grad = True

            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                outputs = model(inputs).squeeze(-1).squeeze(-1)

            losses = self.loss_func_sample(outputs, targets)
            all_losses.extend(losses.tolist())

            pred_confs, _ = torch.max(outputs, dim=1)
            target_confs = outputs[torch.arange(outputs.shape[0]), targets]
            
            # CRITICAL FIX: Clone outputs before modifying inplace to avoid autograd errors during backward
            outputs_clone = outputs.clone()
            outputs_clone[torch.arange(outputs_clone.shape[0]), targets] = float('-inf')
            pred_confs, _ = torch.max(outputs_clone, dim=1)
            m = target_confs - pred_confs
            all_confs.extend(m.tolist())

            # COMPUTE CSG: Get gradient of the sum of losses w.r.t inputs
            loss_sum = losses.sum()
            input_grads = torch.autograd.grad(loss_sum, inputs, create_graph=False, retain_graph=True)[0]
            input_grad_norm = input_grads.flatten(start_dim=1).norm(p=2, dim=1)
            all_csg.extend(input_grad_norm.tolist())

            losses.mean().backward()
            
            if getattr(args, 'track_grad_norms', False) or getattr(args, 'clip_norm', False):
                batch_grads = [p.grad_sample.view(p.grad_sample.size(0), -1) for p in self.training_params]
                batch_norms = torch.cat(batch_grads, dim=1).norm(dim=1)
                all_norms.append(batch_norms)

        if getattr(args, 'clip_norm', False) or getattr(args, 'track_grad_norms', False):
            all_norms = torch.cat(all_norms, dim=0)
            if getattr(args, 'clip_norm', False):
                all_norms = all_norms.clamp(max=args.clip_norm)
            all_norms = all_norms.tolist()
            
        return all_losses, all_confs, all_norms, all_csg


    def train_epoch(self, model, epoch, computed_losses, computed_confidences, grad_norms, computed_csg, args):
        print(f'\n{args.exp_id}-Epoch: %d' % epoch)
        model.train()
        total = 0
        correct = 0
        t0 = time.time()

        # collect init losses
        if (args.track_computed_loss or args.track_confidences or getattr(args, 'track_grad_norms', False)) and epoch == 0:
            losses, confs, norms, csgs = self.get_all_losses(model, args)
            if args.track_computed_loss:
                computed_losses.append(losses)
                computed_csg.append(csgs)
            if args.track_confidences:
                computed_confidences.append(confs)
            if getattr(args, 'track_grad_norms', False):
                grad_norms.append(norms)

        for batch_idx, (inputs, targets, indices) in enumerate(self.trainloader):
            model.zero_grad()
            self.optimizer.zero_grad()
            if isinstance(self.optimizer, DPOptimizer):
                self.optimizer.expected_batch_size = inputs.shape[0]

            inputs, targets = inputs.to(self.device), targets.to(self.device)

            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                outputs = model(inputs).squeeze(-1).squeeze(-1)

            losses = self.loss_func_sample(outputs, targets)
            if getattr(args, 'track_free_loss', False):
                self.free_train_losses.loc[indices.tolist(), epoch] = losses.tolist()

            losses.mean().backward()
            self.optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            minibatch_correct = predicted.eq(targets.data).float().cpu()
            total += targets.size(0)
            correct += minibatch_correct.sum()

        if (args.track_computed_loss or args.track_confidences or getattr(args, 'track_grad_norms', False)):
            losses, confs, norms, csgs = self.get_all_losses(model, args)
            if args.track_computed_loss:
                computed_losses.append(losses)
                computed_csg.append(csgs)
            if args.track_confidences:
                computed_confidences.append(confs)
            if getattr(args, 'track_grad_norms', False):
                grad_norms.append(norms)

        t1 = time.time()
        self.scheduler.step()
        acc = (100. * float(correct) / float(total)) if total > 0 else 0.0
        print('Time: %d s' % (t1 - t0), 'train acc:', acc, end=' ')
        return acc

    def train_test(self, model, args, model_id):
        self.training_params = self.get_training_params(model)
        self.optimizer = self.get_optimizer(self.training_params, args.lr, args.momentum, args.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, args.epochs)

        if getattr(args, 'private', False):
            privacy_engine = PrivacyEngine()
            model, self.optimizer, self.trainloader = privacy_engine.make_private_with_epsilon(max_grad_norm=args.clip_norm, module=model,
                                                                                     optimizer=self.optimizer,
                                                                                     data_loader=self.trainloader,
                                                                                     target_epsilon=args.target_epsilon,
                                                                                     target_delta=args.target_delta,
                                                                                     epochs=args.epochs)

        elif getattr(args, 'clip_norm', False) or getattr(args, 'noise_multiplier', False):
            privacy_engine = PrivacyEngine()
            model, self.optimizer, self.trainloader = privacy_engine.make_private(
                                                                                max_grad_norm=args.clip_norm if args.clip_norm else 0.0,
                                                                                module=model,
                                                                                optimizer=self.optimizer,
                                                                                data_loader=self.trainloader,
                                                                                noise_multiplier=args.noise_multiplier if getattr(args, 'noise_multiplier', False) else 0.0
                                                                                )

        if getattr(args, 'track_grad_norms', False) and not isinstance(model, GradSampleModule):
            model = GradSampleModule(model)

        computed_losses = []
        computed_confidences = []
        grad_norms = []
        computed_csg = []  # NEW

        if getattr(args, 'track_free_loss', False):
            self.free_train_losses = pd.DataFrame(np.full((len(self.trainloader.dataset), args.epochs), np.nan))
            self.free_train_confidences = pd.DataFrame(np.full((len(self.trainloader.dataset), args.epochs), np.nan))

        num_training = len(self.trainloader.dataset)
        num_test = len(self.testloader.dataset)
        print('Training on: ', num_training, 'Testing on: ', num_test)

        print(f'-------- Training model {model_id} --------')
        print('\n==> Starting training')

        for epoch in range(args.epochs):
            train_acc = self.train_epoch(model, epoch, computed_losses, computed_confidences, grad_norms, computed_csg, args)
            test_acc = self.test(model, args)

            if getattr(args, 'checkpoint', False) and (epoch + 1) % 5 == 0:
                save_model(model, args, self.trainloader, train_acc, test_acc, checkpoint=True, epoch=epoch)

        if getattr(args, 'track_free_loss', False):
            save_free_loss(self.free_train_losses, args)

        if args.track_computed_loss:
            save_tracking_data(computed_losses, args)
            save_tracking_data(computed_csg, args, save_dir="csg_norms") # Save CSG metric

        if args.track_confidences:
            save_tracking_data(computed_confidences, args, save_dir="confidences")

        if getattr(args, 'track_grad_norms', False):
            save_tracking_data(grad_norms, args, save_dir="grad_norms")

        save_model(model, args, self.trainloader, train_acc, test_acc)

    def test(self, model, args):
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets, _indices in self.testloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs).squeeze(-1).squeeze(-1)

                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct_idx = predicted.eq(targets.data).cpu()
                correct += correct_idx.sum()

        acc = 100. * float(correct) / float(total)
        print('test acc:', acc)
        return acc


def save_free_loss(train_loss, args):
    train_dir = os.path.join(STORAGE_DIR, 'free_train_losses')
    os.makedirs(train_dir, exist_ok=True)
    file = args.exp_id

    if getattr(args, 'dual', False):
        file += f'_dual_{args.dual}'
    file += '.pq'

    train_path = os.path.join(train_dir, file)
    train_loss.to_parquet(train_path)


def save_tracking_data(computed_losses, args, save_dir=None):
    if not save_dir:
        outdir = os.path.join(STORAGE_DIR, 'losses')
    else:
        outdir = os.path.join(STORAGE_DIR, save_dir)

    os.makedirs(outdir, exist_ok=True)

    file = args.exp_id
    if getattr(args, 'dual', False):
        file += f'_dual_{args.dual}'
    elif getattr(args, 'shadow_count', False):
        file += f'_shadow_{str(args.shadow_id)}'
    file += '.pq'

    fullpath = os.path.join(outdir, file)
    if os.path.exists(fullpath):
        print("'DUPLICATE' METRIC - OVERWRITING PREVIOUS", file=sys.stderr)

    print(fullpath)
    pd.DataFrame(computed_losses).transpose().to_parquet(fullpath)

def save_model(model, args, trainloader, train_acc, test_acc, checkpoint=False, epoch=None):
    if getattr(args, 'shadow_count', False):
        save_name = 'shadow_' + str(args.shadow_id)
    elif getattr(args, 'dual', False):
        save_name = f'dual_{args.dual}'
    elif getattr(args, 'track_computed_loss', False) or getattr(args, 'track_free_loss', False):
        save_name = 'target'
    else:
        save_name = 'model'

    model_state_dict = model.state_dict()

    dic = {
        'model_state_dict': model_state_dict,
        'hyperparameters': vars(args),
        'trained_on_indices': trainloader.dataset.indices,
        'arch': args.arch,
        'seed': args.seed,
        'dataset': args.dataset,
        'train_acc': train_acc,
        'test_acc': test_acc
    }

    dir = os.path.join(MODEL_DIR, args.exp_id)

    if checkpoint:
        dir = os.path.join(dir, f'checkpoint_before_{epoch + 1}')
    os.makedirs(dir, exist_ok=True)
    fullpath = os.path.join(dir, save_name)
    torch.save(dic, fullpath)
