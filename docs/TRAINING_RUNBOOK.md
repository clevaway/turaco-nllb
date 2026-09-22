# Training Runbook

## Before starting

1. Make a persistent copy of this repository and the raw dataset revision.
2. Confirm the runtime has a CUDA GPU and at least 15 GB VRAM for the default full 600M recipe.
3. Decide one direction. Train `en-wes` and `wes-en` as separate runs.
4. Run the tests and the 64-example overfit check before a full job.

## Install and validate

```bash
python -m pip install -e ".[test]"
pytest -q
python -m compileall -q src benchmark
```

## Full T4 run

```bash
turaco-train --config configs/nllb_600m_t4_en_wes.yaml
```

Expected output tree:

```text
outputs/Turaco-NLLB-mt-en-wes/
  best/
  checkpoints/
  data/
    clean_splits.parquet
    rejected_rows.parquet
    data_audit.json
    cleaning_report.json
    split_manifest.json
  evaluation/
    internal_test_metrics.json
    internal_test_predictions.csv
  resolved_config.yaml
  run_summary.json
```

## Runtime recovery

Checkpoints are retained under the run's `checkpoints` directory. Set `training.resume_from_checkpoint` in a copied YAML config to the exact checkpoint directory and rerun. Do not point it at `best` or an incomplete cloud-synchronization folder.

## If the T4 runs out of memory

Make one change at a time:

1. reduce `training.train_batch_size` from 2 to 1;
2. reduce `training.eval_batch_size` from 4 to 2;
3. reduce source/target lengths from 160/192 to 128/160;
4. use LoRA as an engineering fallback, then label the training method accurately.

Keep gradient accumulation high enough to preserve the intended effective batch size.

## If training is too slow

Run a smaller smoke subset first. Use checkpoint/resume for longer runs, and record the wall time and completed optimizer steps.

## Post-training checks

```bash
turaco-translate \
  --model outputs/Turaco-NLLB-mt-en-wes/best \
  --direction en-wes \
  --text "Please send me the report tomorrow morning."
```

Then:

1. reload on CPU and GPU;
2. inspect 100 randomly selected outputs;
3. inspect every empty, copied, very short, and very long output;
4. run TuracoBench without changing the checkpoint;
5. archive predictions and JSON metrics before drafting claims.
