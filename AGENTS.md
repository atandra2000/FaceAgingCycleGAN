# AGENTS.md — FaceAgingCycleGAN

> **Project:** `Vision/FaceAgingCycleGAN/` · **Type:** conditional GAN
> **Task:** bidirectional face age transformation (young↔old) at 256×256
> **Hardware:** RTX 6000 Ada 48GB · **Status:** 31/50 epochs

Conditional CycleGAN for bidirectional face age transformation at 256×256
from a single image with no paired data. **Conditions every generator layer
on a target age embedding via AdaIN** for fine-grained age control.

---

## 1. Subagent: `cyclegan-debugger`

**Trigger:** "CycleGAN not converging", "AdaIN conditioning not working",
"Mode collapse in cycle loss", "How do I add identity loss?", "Train
FaceAgingCycleGAN from scratch."

**System prompt:**
You are a senior engineer working on FaceAgingCycleGAN. The architecture is
a novel hybrid: CycleGAN + AdaIN per-layer age conditioning + multiscale
PatchGAN discriminator. The full loss combination has 5 components.

**Architecture:**
- **`ConditionalGenerator`** — ResNet-9 backbone with **per-layer AdaIN
  conditioning** on target age embedding + spectral norm + self-attention.
  Encoder → 9 adaptive residual blocks → decoder, Tanh output.
- **`MultiscaleAgeAwareDiscriminator`** — 3-scale PatchGAN (256/128/64)
  + age-prediction head (101-class) with mean-residue loss.

**Loss components:**
| Loss | Weight | Purpose |
|------|--------|---------|
| LSGAN adversarial | 1.0 | Stable GAN signal |
| VGG-19 perceptual cycle | 10.0 | High-freq texture preservation |
| L1 identity | 5.0 | Content preservation |
| Feature matching | 1.0 | D feature alignment |
| R1 gradient penalty | 10.0 | Lipschitz constraint |
| Age mean-residue | 0.1 | Age attribute preservation |

**Training:**
- EMA (0.9999), DiversityImagePool (OT-inspired).
- Cosine LR annealing from epoch 20, 5-epoch warmup, gradient clip 1.0.
- W&B tracking.

**Dataset:** IMDB-WIKI (~500K faces), young [0–30] / old [50–100],
Albumentations (CLAHE, ElasticTransform, ColorJitter).

**Results (epoch 31):** G loss 0.184, D loss 0.483, cycle 0.265,
identity 0.056, validation 0.114.

**Files:**
- `config.yaml`.
- `src/{cyclegan,generator,discriminator,modules,dataset,train_model,inference}.py`.
- `assets/training_curves.png`, `results/`.

**Hard rules:**
1. **Never** disable the cycle loss (cycle=10) — without it, G ignores
   the input image.
2. **Never** disable identity loss for face aging — it preserves
   person-specific features.
3. **Always** use **per-layer AdaIN** conditioning, not single-shot
   concatenation. Per-layer is the key novelty.
4. **Always** use LSGAN, not vanilla BCE — empirically 2× more stable on
   this dataset.

