# SKILLS.md — FaceAgingCycleGAN

> Skills for the conditional CycleGAN with AdaIN age conditioning.

---

## Skill 1: Resume from a checkpoint

```bash
cd Vision/FaceAgingCycleGAN
python src/train_model.py --resume checkpoints/cyclegan_epoch_31.pt \
  --config config.yaml --epochs 50
```

Resumes G_AB, G_BA, D_A, D_B, EMA shadow, optimizer state, LR scheduler.

## Skill 2: Tune the AdaIN conditioning

The age embedding (101-dim, one-hot) is fed into every adaptive residual
block:

```python
# src/modules.py — AdaptiveResBlock
self.age_mlp = nn.Sequential(
    nn.Linear(101, 512),
    nn.ReLU(),
    nn.Linear(512, 2 * channels),  # AdaIN: scale + bias
)
scale, bias = self.age_mlp(age_embed).chunk(2, dim=1)
out = scale * normalized + bias
```

To swap to **class-conditional batch norm** (CCBN) instead of AdaIN,
replace the AdaIN call with `nn.BatchNorm2d` using class-conditional
gamma/beta.

## Skill 3: Add a new loss component

Add to the trainer's `compute_loss()`:

```python
# src/train_model.py
def compute_loss(self, real_A, real_B, age_A, age_B):
    ...
    # Existing losses...
    new_loss = F.l1_loss(fake_B, real_B)  # example: structure loss
    return total_loss + 0.5 * new_loss
```

Always include the loss in `config.yaml:loss_weights` — keep weights
tunable from the config, not hard-coded.

## Skill 4: Fix divergence on RTX 6000 Ada

Symptom: loss explodes after epoch 25.

1. **Lower LR** from `2e-4` to `1e-4`.
2. **Increase R1 weight** from 10 to 20 (stronger Lipschitz).
3. **Add EMA** with decay 0.9999 (already enabled).
4. **Reduce D capacity** by halving the channel width at scale 64.
5. **Clip gradients** to 1.0 (already enabled).

## Skill 5: Generate a single image

```bash
python src/inference.py --input input.jpg --target_age 65 \
  --output aged.png --generator_AB checkpoints/G_AB_epoch_31.pt
```

Target age is an int in [0, 100].

## Skill 6: Switch to WGAN-LP (instead of LSGAN)

```python
# src/train_model.py
def d_loss_wgan_lp(d_real, d_fake):
    gp = ((d_real - 1)**2 + d_fake**2).mean()  # Least-squares penalty
    return (d_fake - d_real).mean() + 10.0 * gp
```

WGAN-LP converges faster on small datasets but is sensitive to LR.

## Pitfalls
- **`CycleGAN` requires both directions** (A→B and B→A). Training only
  one direction produces garbage.
- **Identity loss** weight 5.0 is for face aging. Lower it to 1.0 for
  style transfer (the identity should change).
- **`Albumentations` order** matters: Geometric → Color → Noise.
- **`DiversityImagePool`** size 50 — the OT-inspired pool reduces mode
  collapse. Do not reduce below 25.

