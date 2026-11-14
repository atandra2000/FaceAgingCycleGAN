"""Smoke test for FaceAgingCycleGAN Generator and Discriminator."""
import torch
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from generator import ConditionalGenerator
from discriminator import MultiscaleAgeAwareDiscriminator


def test_generator_and_discriminator_shapes():
    batch_size = 2
    # Fast lightweight config
    gen = ConditionalGenerator(input_nc=3, output_nc=3, ngf=16, n_residual_blocks=2, num_ages=10)
    disc = MultiscaleAgeAwareDiscriminator(input_nc=3, ndf=16, n_layers=2, num_ages=10, num_scales=2)
    gen.eval()
    disc.eval()

    # Generator forward
    x = torch.randn(batch_size, 3, 64, 64)
    target_age = torch.randint(0, 5, (batch_size,))
    fake = gen(x, target_age)
    assert fake.shape == (batch_size, 3, 64, 64)

    # Discriminator forward
    age_tensor = torch.randint(0, 10, (batch_size,))
    results, features, age_pred = disc(fake, age_tensor)
    assert len(results) == 2  # 2 scales
    for out in results:
        assert out.dim() == 4
        assert out.shape[0] == batch_size
    assert age_pred.shape == (batch_size, 10)
