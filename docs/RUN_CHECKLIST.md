# Run Checklist — after pretrain finishes

Do these in order. Each step has a **command**, a **check** (how to know it worked), and any **gotcha**. Assumes `n_classes: 5` is set in `config.yaml` and pretrain has produced `models/cnn_lstm_best.pt`.

---

## 0. Confirm pretrain landed
```bash
python -c "import torch; m=torch.load('models/cnn_lstm_best.pt', weights_only=False)['meta']; print('epoch', m.get('epoch'), '| val_loss', round(m.get('val_loss',0),4), '| classes', m.get('flare_class_order'))"
```
**Check:** prints a `val_loss` and `flare_class_order: ['A','B','C','M','X']`. If `flare_class_order` is `None`, the model was trained binary — re-check `config.yaml n_classes: 5` and rerun pretrain.

---

## 1. (Already done) SoLEXS calibration
`config.yaml` has `solexs_counts_to_wm2: 1.806e-08` — calibrated from 435 matched GOES/SoLEXS flare pairs via `scripts/calibrate_solexs.py`. The catalog has been rebuilt with these correct class labels. **Skip this step** unless you add new data and need to recalibrate.

---

## 2. Fine-tune on SoLEXS
```bash
python model/train.py --phase finetune --sxr-dir data/processed --epochs 10
```
**Check:** logs show `Phase 2: Fine-tuning`, a `classes: A:.. B:.. C:.. M:.. X:..` line, falling `vl_loss`, and `saved to models/cnn_lstm_best.pt`. CNN layers are frozen (expected); only LSTM + heads adapt.
**Gotcha:** this overwrites the pretrain checkpoint at the same path. To keep both, pass `--ckpt models/cnn_lstm_finetuned.pt` (and use that path in step 3/4).

---

## 3. Evaluate on held-out days
```bash
python scripts/evaluate_model.py --holdout-frac 0.2
```
**Check:** prints the threshold sweep with **TSS** and **HSS** columns plus lead-time stats (median/p90 lead minutes, false-alarm rate). Sanity: TSS should be clearly > 0; AUC > 0.5. TSS near 0 = no skill → revisit labels/calibration.
**Gotcha:** held-out = most-recent 20% of days (time-based split), so it never tests on trained days. Add `--ckpt <path>` if you saved the fine-tuned model separately.

---

## 4. Launch the dashboard
```bash
C:\Users\HP\AppData\Local\Programs\Python\Python312\python.exe dashboard/app.py
```
Visit `http://127.0.0.1:8050`.
**Check:** pick a date that has data (2024-02-01 → 2026-09-08), instrument = SoLEXS. You should see the SXR light curve with flare markers, the forecast-probability trace, and — because `n_classes: 5` — the **"Flare Intensity-Class Forecast"** panel + "most likely class" metric.
**Gotcha:** requires Python 3.12 (torch is installed there, not in the default `python`). If the class panel doesn't appear, the loaded checkpoint is binary (see step 0). If it warns about missing normalization stats, the checkpoint predates the norm-persistence code — retrain.

---

## Quick reference (fastest path to a demo)
```bash
python model/train.py --phase finetune --epochs 10                                            # 2
python scripts/evaluate_model.py --holdout-frac 0.2                                           # 3
C:\Users\HP\AppData\Local\Programs\Python\Python312\python.exe dashboard/app.py              # 4
```

## Where to run what
- **CPU here:** fine-tune (small, ~823 days) and eval are fine; dashboard is fine.
- **Your GPU box:** any *re*-pretrain, or fine-tuning with many epochs.
