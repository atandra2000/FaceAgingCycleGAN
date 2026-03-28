"""Industry-standard CycleGAN for Face Aging with LSGAN, perceptual losses, multi-scale support, and xAI integration."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19, VGG19_Weights  # For perceptual loss

# Imports from required modules
from generator import ConditionalGenerator  # From generator.py
from discriminator import MultiscaleAgeAwareDiscriminator  # From discriminator.py

class EMAModel:
    """Exponential Moving Average model for generator stabilization.

    Uses per-parameter tensor cloning instead of copy.deepcopy to avoid
    RuntimeError with modules that carry spectral_norm / weight_norm hooks
    (deepcopy is unsupported for those in recent PyTorch versions).
    """

    def __init__(self, model, decay=0.999):
        """
        Args:
            model: The model to track with EMA
            decay: EMA decay coefficient (0.999 recommended)
        """
        self.model = model
        self.decay = decay
        # Clone parameter values — no deepcopy of the module object itself
        self._ema_params = {
            name: param.data.clone().detach()
            for name, param in model.named_parameters()
        }

    def update(self):
        """Update EMA weights."""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                self._ema_params[name].mul_(self.decay).add_(
                    param.data, alpha=1.0 - self.decay
                )

    def apply_shadow(self):
        """Copy EMA weights into the live model for inference."""
        for name, param in self.model.named_parameters():
            param.data.copy_(self._ema_params[name])

    def state_dict(self):
        """Get EMA parameter dict for checkpointing."""
        return {k: v.clone() for k, v in self._ema_params.items()}

    def load_state_dict(self, state_dict):
        """Load EMA parameter dict from checkpoint."""
        for name in self._ema_params:
            if name in state_dict:
                self._ema_params[name].copy_(state_dict[name])

class DiversityImagePool:
    """Image pool with optimal transport approximation for diversity."""
    def __init__(self, pool_size=50, diversity_threshold=0.8):
        self.pool_size = pool_size
        self.diversity_threshold = diversity_threshold
        if self.pool_size > 0:
            self.images = []

    def _wasserstein_approx(self, dist1, dist2):
        """Simple 1D Wasserstein (Earth Mover's) approximation via cumulative sums."""
        cdf1 = torch.cumsum(dist1, dim=0)
        cdf2 = torch.cumsum(dist2, dim=0)
        return torch.mean(torch.abs(cdf1 - cdf2))

    def query(self, images):
        """Return images from pool with diversity check."""
        if self.pool_size == 0:
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.detach(), 0)  # Detach to prevent gradient leak
            if len(self.images) < self.pool_size:
                self.images.append(image)
                return_images.append(image)
            else:
                p = torch.rand(1).item()
                if p > 0.5:
                    random_id = torch.randint(0, self.pool_size, (1,)).item()
                    # OT-inspired diversity check (flatten to 1D dist)
                    flat_img = image.flatten().softmax(0)
                    flat_pool = self.images[random_id].flatten().softmax(0)
                    dist = self._wasserstein_approx(flat_img, flat_pool)
                    if dist > self.diversity_threshold:  # Higher dist means more diverse
                        tmp = self.images[random_id].clone()
                        self.images[random_id] = image
                        return_images.append(tmp)
                    else:
                        return_images.append(image)
                else:
                    return_images.append(image)
        return torch.cat(return_images, 0) if return_images else images

class FaceAgingCycleGAN(nn.Module):
    """CycleGAN for bidirectional face aging with xAI."""
    def __init__(self, input_nc=3, output_nc=3, ngf=64, ndf=64, n_residual_blocks=10,
                 num_ages=101, pool_size=50, diversity_threshold=0.8):
        super(FaceAgingCycleGAN, self).__init__()
        # Generators
        self.G_Y2O = ConditionalGenerator(input_nc, output_nc, ngf, n_residual_blocks, num_ages)
        self.G_O2Y = ConditionalGenerator(input_nc, output_nc, ngf, n_residual_blocks, num_ages)

        # Discriminators
        self.D_Y = MultiscaleAgeAwareDiscriminator(input_nc, ndf, num_ages=num_ages)
        self.D_O = MultiscaleAgeAwareDiscriminator(input_nc, ndf, num_ages=num_ages)

        # Image pools
        self.fake_Y_pool = DiversityImagePool(pool_size, diversity_threshold)
        self.fake_O_pool = DiversityImagePool(pool_size, diversity_threshold)

        # Simple pre-trained age estimator (VGG-based for xAI)
        self.age_estimator = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features[:20].eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.age_head = nn.Linear(512 * 7 * 7, 1).to(device)  # Regression output
        for p in self.age_estimator.parameters():
            p.requires_grad = False  # Freeze to avoid training leakage
        for p in self.age_head.parameters():
            p.requires_grad = False  # Freeze age head for xAI


    def forward(self, young_imgs, old_imgs, young_ages=None, old_ages=None):
        """Forward pass with optional conditioning."""
        results = {}
        batch_size = young_imgs.size(0)
        device = young_imgs.device
        
        # Default ages (non-overlap: young 0-50, old 51-100)
        if young_ages is None:
            young_ages = torch.full((batch_size,), 25, dtype=torch.long, device=device)  # 0–50 midpoint
        if old_ages is None:
            old_ages = torch.full((batch_size,), 75, dtype=torch.long, device=device)    # 51–100 midpoint

        # Image Generation
        fake_old = self.G_Y2O(young_imgs, old_ages)
        fake_young = self.G_O2Y(old_imgs, young_ages)
        
        # Cycle Consistency
        rec_young = self.G_O2Y(fake_old, young_ages)
        rec_old = self.G_Y2O(fake_young, old_ages)

        # Identity
        same_old = self.G_Y2O(old_imgs, old_ages)
        same_young = self.G_O2Y(young_imgs, young_ages)

        results.update({
            'real_young': young_imgs, 'real_old': old_imgs,
            'fake_young': fake_young, 'fake_old': fake_old,
            'rec_young': rec_young, 'rec_old': rec_old,
            'same_young': same_young, 'same_old': same_old
        })
        return results

    def estimate_age(self, image):
        """xAI-based age estimation with saliency map."""
        with torch.set_grad_enabled(True):
            self.age_estimator.eval()
            self.age_head.eval()
            image.requires_grad = True
            # Forward pass
            feats = self.age_estimator(image)
            feats = F.adaptive_avg_pool2d(feats, (7, 7)).flatten(1)
            pred_age = self.age_head(feats)
            # Calculate saliency
            pred_age.mean().backward()
            saliency = image.grad.abs().max(dim=1)[0]
        
            return pred_age.detach().cpu().numpy(), saliency.detach().cpu().numpy()
       

class FaceAgingLoss(nn.Module):
    """Advanced losses with adaptive weighting, perceptual, and mean-residue age loss."""
    
    def __init__(self, lambda_cycle=10.0, lambda_identity=2.0, lambda_age=0.5, 
                 lambda_fm=5.0, lambda_perc=10.0, lambda_gp=0.01, device='cuda'):
        super(FaceAgingLoss, self).__init__()
        
        # Base lambdas for losses
        self.base_lambdas = {
            'cycle': lambda_cycle,      # 10.0
            'identity': lambda_identity,  # 2.0
            'age': lambda_age,           # 0.5
            'fm': lambda_fm,             # 5.0
            'perc': lambda_perc,         # 10.0
            'gp': lambda_gp              # 0.01
        }
        
        self.l1loss = nn.L1Loss()
        vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features.eval()
        self.perceptual_model = nn.Sequential(*list(vgg)[:20]).eval()  # Up to conv4_4
        self.perceptual_model.to(device)
        
        for p in self.perceptual_model.parameters():
            p.requires_grad_(False)

    def adaptive_weights(self, loss_dict, epsilon=1e-8):
        """Stable adaptive weighting."""
        weights = {}
        if not loss_dict:
            return weights
        
        # Clamp to prevent extreme values
        total = sum(v.detach().clamp(min=0, max=1e4).item() for v in loss_dict.values())
        
        for k, v in loss_dict.items():
            # Prevent division by zero and extreme weights
            normalized = max(v.detach().clamp(min=0, max=1e4).item(), epsilon)
            weights[k] = float(self.base_lambdas.get(k, 1.0)) * (normalized / (total + epsilon))
        
        return weights

    def lsgan_loss(self, pred, target_is_real):
        """Least-squares GAN loss with clamping."""
        target = torch.ones_like(pred) if target_is_real else torch.zeros_like(pred)
        loss = 0.5 * torch.mean((pred - target) ** 2)
        # Clamp to prevent explosion
        return loss.clamp(max=100.0)

    def perceptual_loss(self, img1, img2):
        """VGG-based perceptual loss for cycle consistency."""
        # Normalize to [0,1] for VGG
        img1_norm = (img1 + 1) / 2
        img2_norm = (img2 + 1) / 2
        feat1 = self.perceptual_model(img1_norm)
        feat2 = self.perceptual_model(img2_norm)
        return self.l1loss(feat1, feat2)

    def feature_matching_loss(self, real_feats, fake_feats):
        """Feature-matching for stability."""
        if not real_feats:
            # Ensure correct device even when empty
            device = fake_feats[0].device if fake_feats is not None and len(fake_feats) > 0 else self.perceptual_model[0].weight.device
            return torch.tensor(0.0, device=device)

        device = real_feats[0].device
        loss = torch.tensor(0.0, device=device)
        for real_f, fake_f in zip(real_feats, fake_feats):
            loss = loss + self.l1loss(fake_f, real_f.detach())
        return loss / len(real_feats) if real_feats else loss


    def mean_residue_loss(self, age_preds, real_ages):
        """Distribution-based mean-residue loss for age."""
        probs = F.softmax(age_preds, dim=1)
        ages = torch.arange(0, age_preds.size(1), device=age_preds.device).float()
        expected = torch.sum(probs * ages, dim=1)
        variance = torch.sum(probs * (ages - expected.unsqueeze(1)) ** 2, dim=1)
        
        mean_loss = self.l1loss(expected, real_ages.float())
        var_loss = torch.mean(variance)
        
        return mean_loss + 0.1 * var_loss  # Balanced

    def r1_penalty(self, real_imgs, real_preds):
        """R1 gradient penalty with stability checks."""
        if self.base_lambdas.get('gp', 0.0) == 0.0:
            return torch.tensor(0.0, device=real_imgs.device)
    
        if not real_imgs.requires_grad:
            real_imgs.requires_grad_(True)
    
        if isinstance(real_preds, (list, tuple)):
            real_preds = real_preds[0]
    
        # Compute gradients
        grads = torch.autograd.grad(
            outputs=real_preds.sum(),
            inputs=real_imgs,
            create_graph=True,
            only_inputs=True,
            allow_unused=True  # Add this to handle AMP scenarios
        )[0]
        
        # Handle case where gradients are None (happens with AMP)
        if grads is None:
            return torch.tensor(0.0, device=real_imgs.device)
        # Clamp gradient norm to prevent explosion
        grad_norm = grads.view(grads.size(0), -1).pow(2).sum(1)
        
        return 0.5 * self.base_lambdas['gp'] * grad_norm

    def compute_generator_loss(self, results, D_fake_y_preds, D_fake_o_preds,
                               D_real_y_feats=None, D_real_o_feats=None,
                               D_fake_y_age=None, D_fake_o_age=None,
                               young_ages=None, old_ages=None,
                               D_fake_y_feats=None, D_fake_o_feats=None):
        """Generator loss with adaptive weighting."""
        loss_dict = {}

        device = results['real_young'].device
        
        # Adversarial loss
        loss_adv_y = torch.stack([self.lsgan_loss(pred, True) for pred in D_fake_y_preds]).mean()
        loss_adv_o = torch.stack([self.lsgan_loss(pred, True) for pred in D_fake_o_preds]).mean()
        loss_dict['adv_y'] = loss_adv_y
        loss_dict['adv_o'] = loss_adv_o
        
        # Cycle consistency perceptual loss
        loss_cycle_y = self.perceptual_loss(results['rec_young'], results['real_young'])
        loss_cycle_o = self.perceptual_loss(results['rec_old'], results['real_old'])
        loss_dict['cycle_y'] = loss_cycle_y
        loss_dict['cycle_o'] = loss_cycle_o
        
        # Identity loss
        loss_identity_y = self.l1loss(results['same_young'], results['real_young'])
        loss_identity_o = self.l1loss(results['same_old'], results['real_old'])
        loss_dict['identity_y'] = loss_identity_y
        loss_dict['identity_o'] = loss_identity_o
        
        # Feature-matching loss
        if (D_real_y_feats is not None and D_fake_y_feats is not None and
            D_real_o_feats is not None and D_fake_o_feats is not None):
            loss_fm_y = self.feature_matching_loss(D_real_y_feats, D_fake_y_feats)
            loss_fm_o = self.feature_matching_loss(D_real_o_feats, D_fake_o_feats)
            loss_dict['fm_y'] = loss_fm_y
            loss_dict['fm_o'] = loss_fm_o
            # Combine for weighting (average for balance)
            loss_dict['fm'] = (loss_fm_y + loss_fm_o) / 2
        
        # Age consistency loss on generated images
        if D_fake_y_age is not None and young_ages is not None:
            loss_age_y = self.mean_residue_loss(D_fake_y_age, young_ages)
        else:
            loss_age_y = torch.tensor(0.0, device=device)
        
        if D_fake_o_age is not None and old_ages is not None:
            loss_age_o = self.mean_residue_loss(D_fake_o_age, old_ages)
        else:
            loss_age_o = torch.tensor(0.0, device=device)
        
        loss_age = (loss_age_y + loss_age_o) * self.base_lambdas['age']
        loss_dict['age'] = loss_age
        
        # Adaptive weighting
        weights = self.adaptive_weights(loss_dict)
        terms = []
        
        for k, v in loss_dict.items():
            w = weights.get(k, 1.0)
            # Only include fm if present
            if k == 'fm' and 'fm' not in loss_dict:
                continue
            terms.append(w * v)
        
        # Final weighted loss
        weighted_loss = torch.stack(terms).sum()
        loss_dict['total'] = weighted_loss
        
        return weighted_loss, loss_dict

    def compute_discriminator_loss(self, real_imgs, real_preds, fake_preds, 
                                   real_ages, age_preds=None):
        """Discriminator loss with multi-scale and mean-residue age."""
        loss_dict = {}
        
        # LSGAN loss
        loss_real = torch.stack([self.lsgan_loss(pred, True) for pred in real_preds]).mean()
        loss_fake = torch.stack([self.lsgan_loss(pred, False) for pred in fake_preds]).mean()
        
        # R1 gradient penalty
        r1 = self.r1_penalty(real_imgs, real_preds)
        
        loss_dict['real'] = loss_real
        loss_dict['fake'] = loss_fake
        loss_dict['r1'] = r1
        
        # Age (mean-residue implementation)
        if age_preds is not None and real_ages is not None and age_preds.shape[0] == real_ages.shape[0]:
            loss_age = self.mean_residue_loss(age_preds, real_ages)
            loss_dict['age'] = loss_age
        
        # Adaptive weighting
        weights = self.adaptive_weights(loss_dict)
        terms = [weights.get(k, 1.0) * v for k, v in loss_dict.items()]
        
        weighted_loss = torch.stack(terms).sum()
        loss_dict['total'] = weighted_loss
        
        return weighted_loss, loss_dict
