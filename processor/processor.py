import logging
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval, R1_mAP_eval_CC


def _empty_cache():
    """torch.cuda.empty_cache() is a no-op (and an error) without a CUDA build."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _build_evaluator(cfg, num_query):
    """Clothes-changing evaluator for PRCC/LTCC.

    ``R1_mAP_eval_CC.compute`` applies the CC rule; ``compute_general`` scores the
    very same features under the Standard / General rule, so no second forward
    pass is needed.
    """
    return R1_mAP_eval_CC(num_query=num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)


def _topk(cmc, k):
    """cmc[k-1] if that rank exists, else the last available rank (no IndexError)."""
    cmc = np.asarray(cmc)
    if cmc.shape[0] == 0:
        return float('nan')
    return float(cmc[min(k, cmc.shape[0]) - 1])


def _fmt(x):
    """Format an optional loss component for the training log."""
    return 'n/a' if x is None else '{:.3f}'.format(x)


def _log_metrics(logger, cmc, mAP, prefix="CC"):
    parts = ["Rank-{:<3}:{:.1%}".format(r, _topk(cmc, r)) for r in [1, 5, 10]]
    logger.info("{} CMC curve, {}".format(prefix, "  ".join(parts)))
    logger.info("{} mAP Acc. :{:.1%}".format(prefix, mAP))


def _run_query_loader(model, loader, evaluator, device):
    for n_iter, (img, pid, cid_list, cloth_id, _, _) in enumerate(loader):
        with torch.no_grad():
            img = img.to(device)
            feat = model(img)
            evaluator.update((feat, pid, cid_list, cloth_id))
    return evaluator


def evaluate(cfg, model, loaders, bundle, device, logger, mode):
    """Run every protocol the dataset supports and log each in its own block.

    ``cfg.TEST.MODE == 'cc'`` keeps only the clothes-changing number (CSCI's
    default for its published tables); ``'both'`` additionally reports the
    Standard / General number:

    * PRCC -> ``CC`` (test/C query vs test/A gallery) and ``Standard``
      (test/B query vs test/A gallery).
    * LTCC -> ``CC`` and ``General`` over the same query/gallery split.
    """
    model.eval()
    report_both = (mode != 'cc')
    results = {}

    if bundle.query_diff is not None:
        # ---------------- PRCC ----------------
        protocols = [('query_diff', 'CC', 'test/C')]
        if report_both:
            protocols.append(('query_same', 'Standard', 'test/B'))

        for key, prefix, query_desc in protocols:
            loader, num_query, _ = loaders[key]
            evaluator = _build_evaluator(cfg, num_query)
            evaluator.reset()
            _run_query_loader(model, loader, evaluator, device)
            cmc, mAP, _, _, _, _, _ = evaluator.compute() if prefix == 'CC' \
                else evaluator.compute_general()
            logger.info("===== PRCC {} setting (query={}) =====".format(prefix, query_desc))
            _log_metrics(logger, cmc, mAP, prefix=prefix)
            results[prefix] = (cmc, mAP)
            _empty_cache()
        return results

    # ---------------- LTCC (single query split) ----------------
    loader, num_query, _ = loaders['query_diff']
    evaluator = _build_evaluator(cfg, num_query)
    evaluator.reset()
    _run_query_loader(model, loader, evaluator, device)

    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("===== LTCC Clothes-Changing setting =====")
    _log_metrics(logger, cmc, mAP, prefix='CC')
    results['CC'] = (cmc, mAP)

    if report_both:
        cmc_g, mAP_g, _, _, _, _, _ = evaluator.compute_general()
        logger.info("===== LTCC General setting =====")
        _log_metrics(logger, cmc_g, mAP_g, prefix='General')
        results['General'] = (cmc_g, mAP_g)

    _empty_cache()
    return results


def select_score(cfg, results):
    """CSCI picks the best checkpoint on the CC protocol (rank1, mAP)."""
    if cfg.TEST.EVAL_PROTOCOL in results:
        return results[cfg.TEST.EVAL_PROTOCOL]
    # fall back to whatever was computed first
    first = next(iter(results.values()))
    return first


def do_train(
        cfg,
        model,
        loaders,
        bundle,
        optimizer,
        scheduler,
        loss_func,
        device
):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    epochs = cfg.SOLVER.MAX_EPOCHS
    device_type = device.type  # "cuda" or "cpu"

    logger = logging.getLogger("reid")
    logger.info('start training')
    logger.info('splits: {}'.format(bundle))

    train_loader = loaders['train'][0]
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    cloth_acc_meter = AverageMeter()

    scaler = torch.GradScaler(device_type) if device.type == "cuda" else None

    best_score = -1.0
    best_rank1 = 0.0
    best_map = 0.0

    for epoch in range(1, epochs+1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        cloth_acc_meter.reset()

        model.train()

        for n_iter, (img, pid, _, cloth_id) in enumerate(train_loader):
            optimizer.zero_grad()

            img = img.to(device)
            target = pid.to(device)
            target_cloth_id = cloth_id.to(device)

            with torch.autocast(device_type=device_type, enabled=(device_type == "cuda")):
                # dual-branch model: F (id), F' (cloth), plain pooled feature and
                # the attention-weighted maps before pooling (for other modules)
                cls_score, feat, cloth_score, cloth_feat, _, _, _ = model(
                    img, target_cloth=target_cloth_id)
                # Convert to float32 for loss computation
                cls_score = cls_score.float()
                feat = feat.float()

            if torch.is_autocast_enabled():
                # keep the softmax heads in fp32 for stable CE
                cls_score = cls_score.float()
                if cloth_score is not None:
                    cloth_score = cloth_score.float()
            loss, parts = loss_func(cls_score, feat, target, target_cloth_id,
                                    cloth_score=cloth_score, cloth_feat=cloth_feat,
                                    return_parts=True)

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            loss_meter.update(loss.item(), img.shape[0])

            with torch.no_grad():
                acc = (cls_score.max(1)[1] == target).float().mean()
                acc_meter.update(acc.item(), 1)
                if cloth_score is not None:
                    cloth_acc = (cloth_score.max(1)[1] == target_cloth_id).float().mean()
                    cloth_acc_meter.update(cloth_acc.item(), 1)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            if(n_iter + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f} "
                            "(id {} | tri {} | cloth {} | dis {}) Acc: {:.1%} (cloth {:.1%}) Lr: {:.2e}"
                            .format(epoch, (n_iter + 1), len(train_loader),
                                    loss_meter.avg,
                                    _fmt(parts['id']), _fmt(parts['triplet']),
                                    _fmt(parts['cloth']), _fmt(parts['disentangle']),
                                    acc_meter.avg, cloth_acc_meter.avg,
                                    scheduler.get_last_lr()[0]))
        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                    .format(epoch, time_per_batch, train_loader.batch_size / time_per_batch))

        scheduler.step()

        if epoch % checkpoint_period == 0:
            torch.save(model.state_dict(),
                 os.path.join(os.path.join(cfg.OUTPUT_DIR, cfg.EXP_NAME), cfg.MODEL.NAME + '_{}.pth'.format(epoch)))

        if epoch % eval_period == 0:
            logger.info("Validation Results - Epoch: {}".format(epoch))
            results = evaluate(cfg, model, loaders, bundle, device, logger, cfg.TEST.MODE)
            cmc, mAP = select_score(cfg, results)
            logger.info("=> Selected protocol '{}' : Rank-1 {:.1%}, mAP {:.1%}".format(
                cfg.TEST.EVAL_PROTOCOL, cmc[0], mAP))
            if cmc[0] > best_score:
                best_score = cmc[0]
                best_rank1, best_map = float(cmc[0]), float(mAP)
                torch.save(model.state_dict(),
                           os.path.join(os.path.join(cfg.OUTPUT_DIR, cfg.EXP_NAME),
                                        cfg.MODEL.NAME + '_best.pth'))
                logger.info("=> Saved best model at epoch {} (Rank-1 {:.1%})".format(epoch, cmc[0]))

    return best_rank1, best_map


def do_inference(
        cfg,
        model,
        loaders,
        bundle,
        device
):
    logger = logging.getLogger("reid")
    logger.info("Enter inferencing")

    model.eval()
    results = evaluate(cfg, model, loaders, bundle, device, logger, cfg.TEST.MODE)

    for name, (cmc, mAP) in results.items():
        logger.info("{} : Rank-1 {:.1%}, Rank-5 {:.1%}, mAP {:.1%}".format(
            name, _topk(cmc, 1), _topk(cmc, 5), mAP))

    cmc, mAP = select_score(cfg, results)
    return _topk(cmc, 1), _topk(cmc, 5)
