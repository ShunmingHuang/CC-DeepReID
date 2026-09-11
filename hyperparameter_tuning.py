import optuna
from optuna.samplers import TPESampler
import logging
import os
import torch
import numpy as np
import random
import argparse
from configs import cfg
from configs.default import cfg as default_cfg
from datasets import make_dataloader
from models import make_model
from solver import build_optimizer, build_lr_scheduler
from losses import make_loss
from processor import do_train
from utils.logger import setup_logger
import os.path as osp
import yacs.config

# Monkey patch for yacs library to fix Windows encoding issue
_original_merge_from_file = yacs.config.CfgNode.merge_from_file

def _patched_merge_from_file(self, cfg_filename):
    with open(cfg_filename, 'r', encoding='utf-8') as f:
        cfg = self.load_cfg(f)
    self.merge_from_other_cfg(cfg)

yacs.config.CfgNode.merge_from_file = _patched_merge_from_file


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def reset_cfg(cfg, config_file):
    """Reset cfg to default and load from config file."""
    # Reset to default
    cfg.defrost()
    cfg.merge_from_file(config_file)
    cfg.freeze()


def apply_trial_params(cfg, trial_params):
    """Apply trial parameters to config."""
    cfg.defrost()

    # Solver parameters
    if 'BASE_LR' in trial_params:
        cfg.SOLVER.BASE_LR = trial_params['BASE_LR']
    if 'WEIGHT_DECAY' in trial_params:
        cfg.SOLVER.WEIGHT_DECAY = trial_params['WEIGHT_DECAY']
    if 'MOMENTUM' in trial_params:
        cfg.SOLVER.MOMENTUM = trial_params['MOMENTUM']
    if 'MARGIN' in trial_params:
        cfg.SOLVER.MARGIN = trial_params['MARGIN']
    if 'WARMUP_EPOCHS' in trial_params:
        cfg.SOLVER.WARMUP_EPOCHS = trial_params['WARMUP_EPOCHS']

    # Model parameters
    if 'ID_LOSS_WEIGHT' in trial_params:
        cfg.MODEL.ID_LOSS_WEIGHT = trial_params['ID_LOSS_WEIGHT']
    if 'TRIPLET_LOSS_WEIGHT' in trial_params:
        cfg.MODEL.TRIPLET_LOSS_WEIGHT = trial_params['TRIPLET_LOSS_WEIGHT']
    if 'LABELSMOOTH' in trial_params:
        cfg.MODEL.LABELSMOOTH = trial_params['LABELSMOOTH']

    # Input parameters
    if 'RE_PROB' in trial_params:
        cfg.INPUT.RE_PROB = trial_params['RE_PROB']
    if 'FLIP_PROB' in trial_params:
        cfg.INPUT.PROB = trial_params['FLIP_PROB']

    cfg.freeze()


def define_search_space(trial):
    """Define hyperparameter search space for Optuna."""
    return {
        # Solver parameters
        'BASE_LR': trial.suggest_float('BASE_LR', 1e-5, 1e-3, log=True),
        'WEIGHT_DECAY': trial.suggest_float('WEIGHT_DECAY', 1e-5, 1e-3, log=True),
        'MOMENTUM': trial.suggest_float('MOMENTUM', 0.8, 0.99),
        'MARGIN': trial.suggest_float('MARGIN', 0.1, 0.5),
        'WARMUP_EPOCHS': trial.suggest_int('WARMUP_EPOCHS', 0, 10),

        # Model parameters
        'ID_LOSS_WEIGHT': trial.suggest_float('ID_LOSS_WEIGHT', 0.5, 2.0),
        'TRIPLET_LOSS_WEIGHT': trial.suggest_float('TRIPLET_LOSS_WEIGHT', 0.5, 2.0),
        'LABELSMOOTH': trial.suggest_categorical('LABELSMOOTH', [True, False]),

        # Input parameters
        'RE_PROB': trial.suggest_float('RE_PROB', 0.0, 0.5),
        'FLIP_PROB': trial.suggest_float('FLIP_PROB', 0.3, 0.7),
    }


def objective(trial, config_file, num_trials, epochs):
    """Optuna objective function for multi-objective optimization."""
    # Generate trial parameters
    trial_params = define_search_space(trial)

    # Create a fresh config for this trial
    from configs.default import cfg as fresh_cfg
    fresh_cfg.defrost()
    fresh_cfg.merge_from_file(config_file)
    fresh_cfg.SOLVER.MAX_EPOCHS = epochs

    # Set random seed for reproducibility
    fresh_cfg.SOLVER.SEED = 42 + trial.number
    set_seed(fresh_cfg.SOLVER.SEED)

    # Apply trial parameters
    apply_trial_params(fresh_cfg, trial_params)

    # Setup logger for this trial (suppress output)
    logger = logging.getLogger("reid.trial")
    logger.setLevel(logging.WARNING)

    # Create output directories
    output_dir = osp.join(fresh_cfg.OUTPUT_DIR, fresh_cfg.EXP_NAME, f"trial_{trial.number}")
    log_dir = osp.join(fresh_cfg.LOG_DIR, fresh_cfg.EXP_NAME, f"trial_{trial.number}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Create dataloaders
    loaders, bundle = make_dataloader(fresh_cfg)

    # Create model
    num_cloth_classes = bundle.num_train_clothes if fresh_cfg.MODEL.DUAL_BRANCH else 0
    model = make_model(fresh_cfg, num_classes=bundle.num_train_pids,
                       num_cloth_classes=num_cloth_classes)

    device = torch.device(fresh_cfg.MODEL.DEVICE)
    if fresh_cfg.MODEL.DEVICE == 'cuda':
        model = model.to(device)

    # Create loss function
    loss_func = make_loss(fresh_cfg, num_classes=bundle.num_train_pids,
                          num_cloth_classes=num_cloth_classes)

    # Create optimizer
    optimizer = build_optimizer(
        model,
        optim=fresh_cfg.SOLVER.OPTIMIZER_NAME.lower() if fresh_cfg.SOLVER.OPTIMIZER_NAME.lower() in ['adam', 'amsgrad', 'sgd', 'rmsprop', 'radam'] else 'adam',
        lr=fresh_cfg.SOLVER.BASE_LR,
        weight_decay=fresh_cfg.SOLVER.WEIGHT_DECAY,
        momentum=fresh_cfg.SOLVER.MOMENTUM
    )

    # Create scheduler
    scheduler = build_lr_scheduler(
        optimizer,
        lr_scheduler=fresh_cfg.SOLVER.LR_SCHEDULER,
        stepsize=fresh_cfg.SOLVER.STEPSIZE,
        gamma=fresh_cfg.SOLVER.GAMMA,
        max_epoch=fresh_cfg.SOLVER.MAX_EPOCHS,
        warmup_epochs=fresh_cfg.SOLVER.WARMUP_EPOCHS,
        warmup_type=fresh_cfg.SOLVER.WARMUP_TYPE
    )

    # Train the model
    rank1, mAP = do_train(
        fresh_cfg,
        model,
        loaders,
        bundle,
        optimizer,
        scheduler,
        loss_func,
        device=device
    )

    # Report intermediate values
    trial.report(mAP, step=epochs)
    trial.report(rank1, step=epochs)

    # Print trial results
    print(f"Trial {trial.number}: mAP={mAP:.4f}, Rank-1={rank1:.4f}")
    print(f"  Params: {trial_params}")

    return mAP, rank1


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter tuning with Optuna")
    parser.add_argument("--config_file", default="configs/dukemtmc.yml", help="Path to config file")
    parser.add_argument("--n_trials", type=int, default=50, help="Number of Optuna trials")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--study_name", default="reid_hyperparameter_tuning", help="Optuna study name")
    parser.add_argument("--output_dir", default="optuna_studies", help="Directory to save Optuna studies")
    args = parser.parse_args()

    # Create output directory for Optuna studies
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Create Optuna study for multi-objective optimization
    study = optuna.create_study(
        directions=['maximize', 'maximize'],  # Maximize both mAP and rank-1
        study_name=args.study_name,
        sampler=TPESampler(seed=42),
        storage=f"sqlite:///{os.path.join(args.output_dir, f'{args.study_name}.db')}",
        load_if_exists=True
    )

    # Optimize
    study.optimize(
        lambda trial: objective(trial, args.config_file, args.n_trials, args.epochs),
        n_trials=args.n_trials,
        show_progress_bar=True
    )

    # Print results
    print("\n" + "="*50)
    print("Hyperparameter Tuning Complete!")
    print("="*50)

    print(f"\nNumber of finished trials: {len(study.trials)}")

    # Get Pareto front
    pareto_trials = study.best_trials
    print(f"\nNumber of Pareto trials: {len(pareto_trials)}")

    print("\nPareto Front (mAP, Rank-1):")
    for i, trial in enumerate(pareto_trials):
        print(f"  Trial {trial.number}: mAP={trial.values[0]:.4f}, Rank-1={trial.values[1]:.4f}")
        print(f"    Params: {trial.params}")

    # Save best trials
    best_trial = max(pareto_trials, key=lambda t: t.values[0] + t.values[1])
    print(f"\nBest Trial (by sum):")
    print(f"  Trial {best_trial.number}: mAP={best_trial.values[0]:.4f}, Rank-1={best_trial.values[1]:.4f}")
    print(f"  Best Parameters:")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")


if __name__ == '__main__':
    main()
