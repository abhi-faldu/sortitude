# Pruned Model Collapse — Diagnosis & Fix

Your `sortitude_model_results.csv` shows the **Pruned 50% + Int8** variant at **0.0% accuracy**, while plain Int8 holds **89.70%**. Here's what happened and how to fix it in Colab.

> Note: the byte-identical size of `sortitude_int8.tflite` and `sortitude_pruned_int8.tflite` (2,719,480 B) is **normal** — magnitude pruning zeros weights but stores them in the same dense tensors, so the `.tflite` is the same size. That is not the bug.

---

## Root cause

The collapse is **not** an evaluation bug — the int8 and pruned+int8 models use the same benchmark code path, and int8 evaluates fine. The pruned model genuinely degenerated. Two compounding reasons:

1. **Under-trained recovery after aggressive pruning.** You jump straight to 50% sparsity (`final_sparsity=0.50`) but only fine-tune for **5 epochs at `lr=1e-5`**. At that learning rate the model can't recover the accuracy lost when half its weights are zeroed — especially in MobileNetV2's depthwise-separable convolutions, which are already lean and don't tolerate heavy pruning.
2. **Int8 quantization finishes off a weak model.** Quantizing an already-degraded, half-sparse model pushes its logits into a near-constant output. A model that predicts one class for everything lands near 0% on a 6-class set — exactly what you see.

The tell: you never checked the pruned model's accuracy **as a float Keras model** before quantizing, so the failure wasn't caught until the final table.

---

## The fix (paste into Colab, replacing Cell 13)

Two changes: (a) **validate the pruned float model before quantizing**, and (b) **recover properly** — gentler schedule, more epochs, slightly higher LR.

```python
# ── Cell 13 (FIXED): Pruning + recovery + validation, THEN quantize ────────────
import tensorflow_model_optimization as tfmot
prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

# Warm-start from the already fine-tuned model. Recover over more steps.
RECOVERY_EPOCHS = 12
num_steps = int(np.ceil(len(train_ds)) * RECOVERY_EPOCHS)

pruning_params = {
    'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=0.0,
        final_sparsity=0.50,
        begin_step=0,
        end_step=num_steps,
    )
}

pruned_model = prune_low_magnitude(model, **pruning_params)
pruned_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),   # was 1e-5 — too low to recover
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)

pruned_model.fit(
    train_ds,
    epochs=RECOVERY_EPOCHS,                                    # was 5 — too short
    validation_data=val_ds,
    callbacks=[
        tfmot.sparsity.keras.UpdatePruningStep(),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=4, restore_best_weights=True),
    ],
)

# Strip wrappers -> plain Keras model with sparse weights
final_pruned = tfmot.sparsity.keras.strip_pruning(pruned_model)

# ── GATE: confirm the FLOAT pruned model recovered BEFORE quantizing ──
_, pruned_float_acc = final_pruned.evaluate(val_ds, verbose=0)
print(f'Pruned FLOAT accuracy: {pruned_float_acc*100:.2f}%')
assert pruned_float_acc > 0.70, (
    'Pruned model did not recover — raise epochs/LR or lower final_sparsity '
    '(try 0.30) before quantizing.')

# Now quantize the recovered, sparse model
converter = tf.lite.TFLiteConverter.from_keras_model(final_pruned)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type  = tf.int8
converter.inference_output_type = tf.int8

tflite_pruned_q = converter.convert()
PRUNED_PATH = f'{MODELS_PATH}/model_pruned_quant.tflite'
with open(PRUNED_PATH, 'wb') as f:
    f.write(tflite_pruned_q)
print(f'Pruned+Quant size: {len(tflite_pruned_q)/1e6:.2f} MB')
```

### If it still collapses
Drop `final_sparsity` to **0.30** first (MobileNetV2 is sensitive), confirm the float gate passes, then push sparsity up. You can also prune **only the classifier head + top conv block** instead of the whole network — pruning the depthwise base is what hurts most.

---

## Sanity check to add to the benchmark (optional)

A constant-prediction model can masquerade as "working." Add this inside `eval_tflite` to flag it:

```python
# after the loop, before return:
from collections import Counter
# (track preds in a list `pred_classes` during the loop)
print('  prediction distribution:', Counter(pred_classes))
```

If one class dominates every prediction, the model collapsed regardless of the headline accuracy number.

---

## For the presentation

You don't have to re-run anything. Either framing works:
- **Honest (recommended):** keep the failed result and present it as a finding — "50% pruning without enough recovery training collapsed the model; quantization alone was the better trade-off." This is exactly the kind of analysis graders reward.
- **Polished:** re-run the fixed Cell 13 in Colab, get a working third data point, and show all three variants succeeding.
