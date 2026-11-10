# Tiny Transformer from Scratch

A character-level language model trained from random weights to continue fictional hospitality handover notes. The corpus and its generator are included, so the entire experiment can run offline after installing the dependencies.

## What the model does

1. **Tokenisation** maps each character in the training corpus to an integer. Spaces and newlines are tokens. A reserved ID represents characters absent from the training alphabet.
2. **Next-character prediction** takes a 128-character window and predicts the character immediately after each position. For the stream `Venue`, a short training pair would be input `Venu` and target `enue`.
3. **Causal attention** projects hidden states into queries, keys and values. Each head scores the keys against its query, masks future positions, then uses softmax to mix the visible values. The two decoder blocks also use learned positional embeddings at the model input, residual connections, layer normalisation, and feed-forward layers.
4. **Optimisation** uses mean cross-entropy, backpropagation, gradient clipping at 1.0, and AdamW at a learning rate of 0.001. The model has 116,224 trainable parameters, four attention heads per block, a width of 64 and a 128-character context.
5. **Generation** samples one character at a time from the predicted distribution, appends it to the prompt, then repeats. For a sequence longer than 128 characters, prediction uses the most recent 128.

All model and training code is in `model.py` and `train.py`. PyTorch supplies tensors, automatic differentiation, basic layers and the optimiser. The attention calculation, causal mask, next-token batches, training loop and generation loop are implemented in the project.

## Data and evaluation

| Split | Fictional venues | Notes | Characters |
| --- | ---: | ---: | ---: |
| Train | 16 | 640 | 159,287 |
| Validation | 3 | 120 | 30,164 |
| Test | 3 | 120 | 30,156 |

The original corpus is synthetic and contains no customer records. Each venue has notes generated from the same ten case templates and four shifts. Venues are disjoint across the splits, so validation and test measure generalisation to new venue names and field values within the same note format. Recurring wording and shared templates make this an easier benchmark than independent human-written notes.

The tokenizer alphabet is built from the training split alone. The validation split contains 40 uppercase Qs and the test split contains 40 uppercase Us absent from that alphabet. They are scored as the reserved unknown token. Training draws random windows from the training stream. At each evaluation, the script scores fixed, non-overlapping windows from the training and validation streams and omits the trailing incomplete window. Each position is scored with its preceding context within that window. The final checkpoint is also scored on the separate test split. Loss is mean cross-entropy in natural-log units per character. Lower is better.

The comparison uses the same two prompts at every checkpoint, with the same seed (2026), temperature (0.8), and generation length (200 characters). One prompt names a training venue and one names a validation venue. Checkpoints are inference snapshots containing the model weights, config, training step, and tokenizer alphabet.

## Reproduce the run

Run these commands from the extracted project directory with Python 3.12. The published run used CPU PyTorch 2.7.1 with four threads and a training seed of 42.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install matplotlib==3.10.8
python build_corpus.py
python train.py --out reproduced
python plot.py reproduced
```

Use a fresh output directory for each training run. Expect small numerical differences across hardware or PyTorch builds. On Windows PowerShell, activate the environment with `.venv\Scripts\Activate.ps1`.

Generate from a checkpoint with a custom prompt, or pass a UTF-8 prompt file with `--prompt-file` for multiline input.

```bash
python generate.py run/step-2400.pt --prompt "Venue: Alder House" --count 200
```

## What changed during training

Validation loss fell from 4.2801 at initialisation to 0.7128 at step 1,800. It finished at 0.7365 while training loss continued falling to 0.2042. The widening gap and late validation plateau show diminishing gains on unseen venues. The final checkpoint scored 0.7059 on the test split. The run took 97.1 seconds on CPU.

At step 100, the sampled text mostly shows fragments of words. By step 1,200, it opens with recognisable phrases. By step 2,400, it produces familiar labels and some complete clauses, but it still blends cases. For example, the validation prompt asks about lost property and the model emits a kitchen status. During evaluation each prediction receives the actual preceding characters. During generation the model conditions on its own output, so mistakes can accumulate.

## Included results

- [Loss chart](run/loss.png) and [raw loss values](run/loss.csv).
- [Checkpoint comparison](SAMPLES.md) and [raw generated continuations](run/samples.jsonl).
- [Run metadata](run/run.json) with test loss, settings, and model size.
- Four loadable checkpoints in `run/`, from initialisation through the final training step.
