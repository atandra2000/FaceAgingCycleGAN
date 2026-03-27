"""
Face Aging CycleGAN Training Script
Optimized for NVIDIA RTX6000Ada with checkpoint resume, proper age loss, and clean logging.
"""

import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import wandb
import lpips
from torch.amp import GradScaler, autocast
from torchmetrics.image.fid import FrechetInceptionDistance

from cyclegan import FaceAgingCycleGAN, FaceAgingLoss, EMAModel, DiversityImagePool
from dataset import FaceAgingDataModule


class FaceAgingTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() and config['device']['use_cuda'] else 'cpu')
        
        # Set random seed
        if 'seed' in config['training']:
            torch.manual_seed(config['training']['seed'])
            np.random.seed(config['training']['seed'])
            if torch.cuda.is_available():
                torch.cuda.manual_seed(config['training']['seed'])
        
        # Setup model
        self.model = FaceAgingCycleGAN(
            input_nc=config['model']['input_nc'],
            output_nc=config['model']['output_nc'],
            ngf=config['model']['ngf'],
            ndf=config['model']['ndf'],
            n_residual_blocks=config['model']['n_residual_blocks'],
            num_ages=config['model']['num_ages'],
        ).to(self.device)
        
        # Setup loss criterion
        self.criterion = FaceAgingLoss(
            lambda_cycle=config['training']['lambda_cycle'],
            lambda_identity=config['training']['lambda_identity'],
            lambda_age=config['training']['lambda_age'],
            lambda_fm=config['training']['lambda_fm'],
            lambda_perc=0.0,
            lambda_gp=config['training']['lambda_gp'],
            device=self.device
        )
        
        # Setup AMP
        self.use_amp = self.config['training'].get('use_amp', False)
        if self.use_amp:
            self.scaler = GradScaler()
        
        # Setup data
        self.data_module = FaceAgingDataModule(self.config)
        ((self.young_train_loader, self.old_train_loader),
         (self.young_val_loader, self.old_val_loader),
         (self.young_test_loader, self.old_test_loader)) = self.data_module.get_domain_dataloaders()
        
        # Setup optimizers and schedulers
        self.setup_optimizers()
        
        # Setup EMA
        self._setup_ema()
        
        # Image pools for discriminators
        self.fake_Y_pool = DiversityImagePool(
            config['training'].get('pool_size', 50),
            config['training'].get('diversity_threshold', 0.8)
        )
        self.fake_O_pool = DiversityImagePool(
            config['training'].get('pool_size', 50),
            config['training'].get('diversity_threshold', 0.8)
        )
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.grad_clip = config['training'].get('grad_clip', None)
        self.diversity_threshold = config['training'].get('diversity_threshold', 0.0)
        self.start_epoch = 1  # For resume support
        
        # Setup logging intervals
        self.log_interval = config['wandb'].get('log_interval', 50)
        self.log_images_interval = config['wandb'].get('log_images_interval', 100)
        
        # Store eval flags
        self.fid_calculation = config['eval'].get('fid_calculation', False)
        self.lpips_calculation = config['eval'].get('lpips_calculation', False)
        
        # WandB setup
        self._setup_wandb()
        
        # Load checkpoint if resume_from is specified
        self._maybe_load_checkpoint()
        
        # Metrics tracking
        self.metrics = {
            'train_loss_G': [],
            'train_loss_D': [],
            'val_loss': [],
            'learning_rates': []
        }
    
    def _setup_wandb(self):
        """Initialize Weights & Biases logging"""
        if self.config['wandb']['enabled']:
            wandb.init(
                project=self.config['wandb']['project'],
                entity=self.config['wandb']['entity'],
                name=self.config['wandb']['run_name'],
                config={
                    'model': self.config['model'],
                    'training': self.config['training'],
                    'data': self.config['data'],
                    'lambda_cycle': self.config['training']['lambda_cycle'],
                    'lambda_identity': self.config['training']['lambda_identity'],
                    'lambda_age': self.config['training']['lambda_age'],
                    'lambda_fm': self.config['training']['lambda_fm'],
                    'lambda_gp': self.config['training']['lambda_gp'],
                },
                tags=self.config['wandb']['tags'],
                notes=self.config['wandb']['notes']
            )
            # Lighter logging (gradients only, not all parameters)
            wandb.watch(self.model, log='gradients', log_freq=self.log_interval)
    
    def _setup_ema(self):
        """Setup EMA models if enabled"""
        self.use_ema = self.config['training'].get('use_ema', False)
        if not self.use_ema:
            self.ema_G_Y2O = None
            self.ema_G_O2Y = None
            return
        
        # Setup EMA for both generators
        ema_decay = self.config['training'].get('ema_decay', 0.999)
        self.ema_G_Y2O = EMAModel(self.model.G_Y2O, decay=ema_decay)
        self.ema_G_O2Y = EMAModel(self.model.G_O2Y, decay=ema_decay)
    
    def setup_optimizers(self):
        """Setup optimizers with proper warmup and LR scheduling"""
        lr_g = self.config['training']['learning_rate']['generator']
        lr_d = self.config['training']['learning_rate']['discriminator']
        betas = tuple(self.config['training']['betas'])
        weight_decay = self.config['training'].get('weight_decay', 0.0)
        
        self.optimizer_G = optim.Adam(
            list(self.model.G_Y2O.parameters()) + list(self.model.G_O2Y.parameters()),
            lr=lr_g, betas=betas, weight_decay=weight_decay
        )
        self.optimizer_D_Y = optim.Adam(
            self.model.D_Y.parameters(),
            lr=lr_d, betas=betas, weight_decay=weight_decay
        )
        self.optimizer_D_O = optim.Adam(
            self.model.D_O.parameters(),
            lr=lr_d, betas=betas, weight_decay=weight_decay
        )
        
        # Learning rate schedulers with warmup
        warmup_epochs = self.config['training'].get('warmup_epochs', 5)
        start_decay = self.config['training']['lr_scheduler'].get('start_epoch', 20)
        total_epochs = self.config['training']['num_epochs']
        min_lr = self.config['training']['lr_scheduler'].get('min_lr', 0.000001)
        scheduler_type = self.config['training']['lr_scheduler'].get('type', 'cosine')
        
        def lr_lambda_with_warmup(epoch):
            # Phase 1: Warmup
            if warmup_epochs > 0 and epoch < warmup_epochs:
                return float(epoch + 1) / float(max(1, warmup_epochs))
            
            # Phase 2: Decay
            elif epoch >= start_decay:
                if scheduler_type == 'cosine':
                    progress = (epoch - start_decay) / max(1, total_epochs - start_decay)
                    cosine_decay = 0.5 * (1.0 + np.cos(np.pi * progress))
                    return max(min_lr / lr_g, cosine_decay)
                else:
                    return max(min_lr / lr_g, 1.0 - (epoch - start_decay) / (total_epochs - start_decay))
            
            # Phase 3: Constant
            else:
                return 1.0
        
        self.scheduler_G = optim.lr_scheduler.LambdaLR(self.optimizer_G, lr_lambda=lr_lambda_with_warmup)
        self.scheduler_D_Y = optim.lr_scheduler.LambdaLR(self.optimizer_D_Y, lr_lambda=lr_lambda_with_warmup)
        self.scheduler_D_O = optim.lr_scheduler.LambdaLR(self.optimizer_D_O, lr_lambda=lr_lambda_with_warmup)
    
    def _maybe_load_checkpoint(self):
        """Load checkpoint if resume_from is specified in config"""
        resume_path = self.config['training'].get('resume_from', None)
        if not resume_path:
            return
        
        if not os.path.isfile(resume_path):
            print(f"⚠️ Resume checkpoint not found at {resume_path}, starting from scratch.")
            return
        
        print(f"🔁 Resuming training from checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=self.device)
        
        # Restore model and optimizers
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        self.optimizer_D_Y.load_state_dict(checkpoint['optimizer_D_Y_state_dict'])
        self.optimizer_D_O.load_state_dict(checkpoint['optimizer_D_O_state_dict'])
        
        # Restore schedulers if present
        if 'scheduler_G_state_dict' in checkpoint:
            self.scheduler_G.load_state_dict(checkpoint['scheduler_G_state_dict'])
            self.scheduler_D_Y.load_state_dict(checkpoint['scheduler_D_Y_state_dict'])
            self.scheduler_D_O.load_state_dict(checkpoint['scheduler_D_O_state_dict'])
        
        # Restore EMA if used
        if self.use_ema and 'ema_G_Y2O_state_dict' in checkpoint:
            self.ema_G_Y2O.load_state_dict(checkpoint['ema_G_Y2O_state_dict'])
            self.ema_G_O2Y.load_state_dict(checkpoint['ema_G_O2Y_state_dict'])
        
        # Epoch / loss state
        self.current_epoch = checkpoint.get('epoch', 0)
        self.start_epoch = self.current_epoch + 1
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.global_step = checkpoint.get('global_step', 0)
        
        print(f"✅ Resumed from epoch {self.current_epoch}, continuing at epoch {self.start_epoch}.")
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        
        epoch_loss_G = 0.0
        epoch_loss_D = 0.0
        
        pbar = tqdm(
            zip(self.young_train_loader, self.old_train_loader),
            desc=f'Epoch {epoch}/{self.config["training"]["num_epochs"]}',
            total=min(len(self.young_train_loader), len(self.old_train_loader)))
        
        for batch_idx, (young_batch, old_batch) in enumerate(pbar):
            # Non-blocking transfer for better GPU utilization
            young_imgs = young_batch['image'].to(self.device, non_blocking=True)
            young_ages = young_batch['age'].to(self.device, non_blocking=True)
            old_imgs = old_batch['image'].to(self.device, non_blocking=True)
            old_ages = old_batch['age'].to(self.device, non_blocking=True)
            
            # ================== Train Discriminators ==================
            self.optimizer_D_Y.zero_grad()
            self.optimizer_D_O.zero_grad()
            
            with autocast(device_type='cuda', enabled=self.use_amp, dtype=torch.float16):
                # Generate fake images
                fake_old = self.model.G_Y2O(young_imgs, old_ages)
                fake_young = self.model.G_O2Y(old_imgs, young_ages)
                
                # Use image pools
                fake_old_pooled = self.fake_O_pool.query(fake_old.detach())
                fake_young_pooled = self.fake_Y_pool.query(fake_young.detach())
                
                # Get discriminator predictions - FIX: Pass ages for fake images too
                real_Y_preds, real_Y_feats, real_Y_age = self.model.D_Y(young_imgs, young_ages)
                fake_Y_preds, fake_Y_feats, fake_Y_age = self.model.D_Y(fake_young_pooled, young_ages)
                
                real_O_preds, real_O_feats, real_O_age = self.model.D_O(old_imgs, old_ages)
                fake_O_preds, fake_O_feats, fake_O_age = self.model.D_O(fake_old_pooled, old_ages)
                
                # Discriminator losses
                loss_D_Y, _ = self.criterion.compute_discriminator_loss(
                    young_imgs, real_Y_preds, fake_Y_preds, young_ages, real_Y_age
                )
                loss_D_O, _ = self.criterion.compute_discriminator_loss(
                    old_imgs, real_O_preds, fake_O_preds, old_ages, real_O_age
                )
                
                loss_D = (loss_D_Y + loss_D_O) * 0.5
            
            if self.use_amp:
                self.scaler.scale(loss_D).backward()
                if self.grad_clip:
                    self.scaler.unscale_(self.optimizer_D_Y)
                    self.scaler.unscale_(self.optimizer_D_O)
                    nn.utils.clip_grad_norm_(self.model.D_Y.parameters(), self.grad_clip)
                    nn.utils.clip_grad_norm_(self.model.D_O.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer_D_Y)
                self.scaler.step(self.optimizer_D_O)
                self.scaler.update()
            else:
                loss_D.backward()
                if self.grad_clip:
                    nn.utils.clip_grad_norm_(self.model.D_Y.parameters(), self.grad_clip)
                    nn.utils.clip_grad_norm_(self.model.D_O.parameters(), self.grad_clip)
                self.optimizer_D_Y.step()
                self.optimizer_D_O.step()
            
            # ================== Train Generators ==================
            self.optimizer_G.zero_grad()
            
            with autocast(device_type='cuda', enabled=self.use_amp, dtype=torch.float16):
                # Forward pass
                results = self.model(young_imgs, old_imgs, young_ages, old_ages)
                
                # Get discriminator predictions for fake images - FIX: Pass ages
                fake_Y_preds, fake_Y_feats, fake_Y_age = self.model.D_Y(results['fake_young'], young_ages)
                fake_O_preds, fake_O_feats, fake_O_age = self.model.D_O(results['fake_old'], old_ages)
                
                # Get real features for feature matching
                real_Y_preds, real_Y_feats, real_Y_age = self.model.D_Y(young_imgs, young_ages)
                real_O_preds, real_O_feats, real_O_age = self.model.D_O(old_imgs, old_ages)
                
                loss_G, loss_G_dict = self.criterion.compute_generator_loss(
                    results=results,
                    D_fake_y_preds=fake_Y_preds,
                    D_fake_o_preds=fake_O_preds,
                    D_real_y_feats=real_Y_feats,
                    D_real_o_feats=real_O_feats,
                    D_fake_y_age=fake_Y_age,
                    D_fake_o_age=fake_O_age,
                    young_ages=young_ages,
                    old_ages=old_ages,
                    D_fake_y_feats=fake_Y_feats,
                    D_fake_o_feats=fake_O_feats
                )
                
                # Diversity loss
                if self.diversity_threshold > 0:
                    diversity_penalty = self._compute_diversity_penalty(
                        results['fake_old'], results['fake_young']
                    )
                    if diversity_penalty < self.diversity_threshold:
                        loss_G += (self.diversity_threshold - diversity_penalty) * 0.1
            
            if self.use_amp:
                self.scaler.scale(loss_G).backward()
                if self.grad_clip:
                    self.scaler.unscale_(self.optimizer_G)
                    torch.nn.utils.clip_grad_norm_(
                        list(self.model.G_Y2O.parameters()) + list(self.model.G_O2Y.parameters()),
                        self.grad_clip
                    )
                self.scaler.step(self.optimizer_G)
                self.scaler.update()
            else:
                loss_G.backward()
                if self.grad_clip:
                    nn.utils.clip_grad_norm_(
                        list(self.model.G_Y2O.parameters()) + list(self.model.G_O2Y.parameters()),
                        self.grad_clip
                    )
                self.optimizer_G.step()
            
            # Update EMA
            if self.use_ema:
                self.ema_G_Y2O.update()
                self.ema_G_O2Y.update()
            
            # Track metrics
            epoch_loss_G += loss_G.item()
            epoch_loss_D += loss_D.item()
            
            if batch_idx % self.log_interval == 0:
                self._log_training_metrics(
                    epoch, batch_idx, loss_G.item(), loss_D.item(), loss_G_dict
                )
            
            # Log sample images
            if batch_idx % self.log_images_interval == 0:
                self._save_and_log_samples(results, epoch, batch_idx)
            
            if batch_idx % 20 == 0:
                pbar.set_postfix(
                    lossG=f"{loss_G.item():.4f}",
                    lossD=f"{loss_D.item():.4f}",
                    refresh=True)
            else:
                pbar.set_postfix(
                    lossG=f"{loss_G.item():.4f}",
                    lossD=f"{loss_D.item():.4f}",
                    refresh=False)
            
            self.global_step += 1
        
        num_batches = min(len(self.young_train_loader), len(self.old_train_loader))
        return epoch_loss_G / num_batches, epoch_loss_D / num_batches
    
    def _compute_diversity_penalty(self, fake_old, fake_young):
        """Compute diversity metric for generated images"""
        diversity = (fake_old.std() + fake_young.std()) / 2
        return diversity.item()
    
    def _log_training_metrics(self, epoch, batch_idx, loss_G, loss_D, loss_dict):
        """Log training metrics to WandB with proper aggregation"""
        if self.config['wandb']['enabled']:
            # Safely aggregate split losses
            cycle_total = 0.0
            if 'cycle_y' in loss_dict or 'cycle_o' in loss_dict:
                cycle_total = float(loss_dict.get('cycle_y', 0.0)) + float(loss_dict.get('cycle_o', 0.0))
            
            identity_total = 0.0
            if 'identity_y' in loss_dict or 'identity_o' in loss_dict:
                identity_total = float(loss_dict.get('identity_y', 0.0)) + float(loss_dict.get('identity_o', 0.0))
            
            age_total = float(loss_dict.get('age', 0.0))
            fm_total = float(loss_dict.get('fm', 0.0))
            gp_total = float(loss_dict.get('gp', 0.0))
            
            metrics = {
                'train/loss_G': loss_G,
                'train/loss_D': loss_D,
                'train/loss_cycle': cycle_total,
                'train/loss_identity': identity_total,
                'train/loss_age': age_total,
                'train/loss_fm': fm_total,
                'train/loss_gp': gp_total,
                'train/lr_G': self.optimizer_G.param_groups[0]['lr'],
                'train/lr_D': self.optimizer_D_Y.param_groups[0]['lr'],
                'epoch': epoch,
                'step': self.global_step
            }
            wandb.log(metrics)
    
    def _save_and_log_samples(self, results, epoch, batch_idx):
        """Save and log sample images to WandB"""
        if self.config['wandb']['enabled']:
            images = {
                'young_real': wandb.Image(results['real_young'][0]),
                'old_real': wandb.Image(results['real_old'][0]),
                'young_to_old': wandb.Image(results['fake_old'][0]),
                'old_to_young': wandb.Image(results['fake_young'][0]),
                'young_reconstructed': wandb.Image(results['rec_young'][0]),
                'old_reconstructed': wandb.Image(results['rec_old'][0]),
            }
            wandb.log({'samples': images, 'step': self.global_step})
    
    @torch.no_grad()
    def validate(self, epoch):
        """Validate the model"""
        self.model.eval()
        val_loss = 0.0
    
        # Setup metrics if needed
        if self.fid_calculation or self.lpips_calculation:
            try:
                if self.fid_calculation:
                    fid_metric = FrechetInceptionDistance(feature=2048).to(self.device)
                if self.lpips_calculation:
                    lpips_metric = lpips.LPIPS(net='alex').to(self.device)
            except ImportError:
                print("⚠️ FID/LPIPS libraries not available, skipping metrics")
                self.fid_calculation = False
                self.lpips_calculation = False
    
        pbar = tqdm(
            zip(self.young_val_loader, self.old_val_loader),
            desc='Validation',
            total=min(len(self.young_val_loader), len(self.old_val_loader)))
    
        num_batches = 0
        lpips_scores = []  # Accumulate LPIPS scores
    
        for young_batch, old_batch in pbar:
            young_imgs = young_batch['image'].to(self.device, non_blocking=True)
            young_ages = young_batch['age'].to(self.device, non_blocking=True)
            old_imgs = old_batch['image'].to(self.device, non_blocking=True)
            old_ages = old_batch['age'].to(self.device, non_blocking=True)
        
            # Skip batches with mismatched sizes
            if young_imgs.size(0) != old_imgs.size(0):
                continue
        
            # Minimum batch size to avoid instability
            if young_imgs.size(0) < 2:
                continue
        
            try:
                results = self.model(young_imgs, old_imgs, young_ages, old_ages)
            
                # Get discriminator predictions
                fake_Y_preds, fake_Y_feats, fake_Y_age = self.model.D_Y(results['fake_young'], young_ages)
                fake_O_preds, fake_O_feats, fake_O_age = self.model.D_O(results['fake_old'], old_ages)
                real_Y_preds, real_Y_feats, real_Y_age = self.model.D_Y(young_imgs, young_ages)
                real_O_preds, real_O_feats, real_O_age = self.model.D_O(old_imgs, old_ages)
            
                loss_G, loss_G_dict = self.criterion.compute_generator_loss(
                    results=results,
                    D_fake_y_preds=fake_Y_preds,
                    D_fake_o_preds=fake_O_preds,
                    D_real_y_feats=real_Y_feats,
                    D_real_o_feats=real_O_feats,
                    D_fake_y_age=fake_Y_age,
                    D_fake_o_age=fake_O_age,
                    young_ages=young_ages,
                    old_ages=old_ages,
                    D_fake_y_feats=fake_Y_feats,
                    D_fake_o_feats=fake_O_feats
                )
            
                val_loss += loss_G.item()
                num_batches += 1
            
                # Compute FID if enabled
                if self.fid_calculation:
                    fid_metric.update(young_imgs, real=True)
                    fid_metric.update(results['fake_young'], real=False)
            
                # Compute LPIPS if enabled
                if self.lpips_calculation:
                    lpips_score = lpips_metric(young_imgs, results['rec_young'])
                    lpips_scores.append(lpips_score.mean().item())
        
            except Exception as e:
                print(f"⚠️ Validation batch failed: {e}, skipping...")
                continue
    
        if num_batches == 0:
            print("⚠️ No valid validation batches! Returning inf loss.")
            return float('inf')
    
        avg_val_loss = val_loss / num_batches
    
        # Log validation metrics
        if self.config['wandb']['enabled']:
            metrics = {'val/loss': avg_val_loss, 'epoch': epoch}
        
            if self.fid_calculation:
                try:
                    fid_score = fid_metric.compute()
                    metrics['val/fid'] = fid_score
                    fid_metric.reset()
                except:
                    print("⚠️ FID computation failed, skipping")
        
            if self.lpips_calculation and lpips_scores:
                metrics['val/lpips'] = sum(lpips_scores) / len(lpips_scores)
        
            wandb.log(metrics)
    
        return avg_val_loss

    @torch.no_grad()
    def test(self):
        """Test the model on test set"""
        self.model.eval()
        test_loss = 0.0

        print("Starting testing on test set...")
        # Setup metrics if needed
        if self.fid_calculation or self.lpips_calculation:
            try:
                if self.fid_calculation:
                    fid_metric = FrechetInceptionDistance(feature=2048).to(self.device)
                if self.lpips_calculation:
                    lpips_metric = lpips.LPIPS(net='alex').to(self.device)
            except ImportError:
                print("⚠️ FID/LPIPS libraries not available, skipping metrics")
                self.fid_calculation = False
                self.lpips_calculation = False
    
        pbar = tqdm(
            zip(self.young_test_loader, self.old_test_loader),
            desc='Testing',
            total=min(len(self.young_test_loader), len(self.old_test_loader)))
    
        num_batches = 0
        lpips_scores = []
    
        for young_batch, old_batch in pbar:
            young_imgs = young_batch['image'].to(self.device, non_blocking=True)
            young_ages = young_batch['age'].to(self.device, non_blocking=True)
            old_imgs = old_batch['image'].to(self.device, non_blocking=True)
            old_ages = old_batch['age'].to(self.device, non_blocking=True)
        
            # Skip batches with mismatched sizes
            if young_imgs.size(0) != old_imgs.size(0):
                continue
        
            # Skip very small batches
            if young_imgs.size(0) < 2:
                continue
        
            try:
                results = self.model(young_imgs, old_imgs, young_ages, old_ages)
            
                # Get discriminator predictions
                fake_Y_preds, fake_Y_feats, fake_Y_age = self.model.D_Y(results['fake_young'], young_ages)
                fake_O_preds, fake_O_feats, fake_O_age = self.model.D_O(results['fake_old'], old_ages)
                real_Y_preds, real_Y_feats, real_Y_age = self.model.D_Y(young_imgs, young_ages)
                real_O_preds, real_O_feats, real_O_age = self.model.D_O(old_imgs, old_ages)
            
                loss_G, loss_G_dict = self.criterion.compute_generator_loss(
                    results=results,
                    D_fake_y_preds=fake_Y_preds,
                    D_fake_o_preds=fake_O_preds,
                    D_real_y_feats=real_Y_feats,
                    D_real_o_feats=real_O_feats,
                    D_fake_y_age=fake_Y_age,
                    D_fake_o_age=fake_O_age,
                    young_ages=young_ages,
                    old_ages=old_ages,
                    D_fake_y_feats=fake_Y_feats,
                    D_fake_o_feats=fake_O_feats
                )
            
                test_loss += loss_G.item()
                num_batches += 1
            
                # Compute FID if enabled
                if self.fid_calculation:
                    fid_metric.update(young_imgs, real=True)
                    fid_metric.update(results['fake_young'], real=False)
            
                # Compute LPIPS if enabled
                if self.lpips_calculation:
                    lpips_score = lpips_metric(young_imgs, results['rec_young'])
                    lpips_scores.append(lpips_score.mean().item())
        
            except Exception as e:
                print(f"⚠️ Test batch failed: {e}, skipping...")
                continue
    
        if num_batches == 0:
            print("⚠️ No valid test batches!")
            return float('inf')
    
        avg_test_loss = test_loss / num_batches
        print(f"Test Loss: {avg_test_loss:.4f}")
    
        # Log test metrics
        if self.config['wandb']['enabled']:
            metrics = {'test/loss': avg_test_loss}
        
            if self.fid_calculation:
                try:
                    fid_score = fid_metric.compute()
                    metrics['test/fid'] = fid_score
                    print(f"Test FID: {fid_score:.4f}")
                    fid_metric.reset()
                except:
                    print("⚠️ FID computation failed")
        
            if self.lpips_calculation and lpips_scores:
                avg_lpips = sum(lpips_scores) / len(lpips_scores)
                metrics['test/lpips'] = avg_lpips
                print(f"Test LPIPS: {avg_lpips:.4f}")
        
            wandb.log(metrics)
    
        return avg_test_loss

    def save_checkpoint(self, filename):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'optimizer_D_Y_state_dict': self.optimizer_D_Y.state_dict(),
            'optimizer_D_O_state_dict': self.optimizer_D_O.state_dict(),
            'scheduler_G_state_dict': self.scheduler_G.state_dict(),
            'scheduler_D_Y_state_dict': self.scheduler_D_Y.state_dict(),
            'scheduler_D_O_state_dict': self.scheduler_D_O.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        if self.use_ema:
            checkpoint['ema_G_Y2O_state_dict'] = self.ema_G_Y2O.state_dict()
            checkpoint['ema_G_O2Y_state_dict'] = self.ema_G_O2Y.state_dict()
        
        checkpoint_dir = self.config['training']['checkpoint_dir']
        os.makedirs(checkpoint_dir, exist_ok=True)
        filepath = os.path.join(checkpoint_dir, filename)
        torch.save(checkpoint, filepath)
        print(f"💾 Checkpoint saved: {filepath}")
    
    def train(self):
        """Main training loop"""
        print(f"Starting from epoch: {self.start_epoch}")
        
        for epoch in range(self.start_epoch, self.config['training']['num_epochs'] + 1):
            self.current_epoch = epoch
            
            # Train
            train_loss_G, train_loss_D = self.train_epoch(epoch)
            
            # Save checkpoint at specified frequency
            if epoch % self.config['training']['save_freq'] == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pth')
            
            # Validate
            if epoch % self.config['eval']['every_n_epochs'] == 0:
                try:
                    val_loss = self.validate(epoch)
                
                    # Save best model
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.save_checkpoint('best_model.pth')
                        print(f"✅ New best model saved (val_loss: {val_loss:.4f})")
                except Exception as e:
                    print(f"⚠️ Validation failed with error: {e}")
            
            # Step schedulers
            self.scheduler_G.step()
            self.scheduler_D_Y.step()
            self.scheduler_D_O.step()
            
            # Save checkpoint
            if epoch % self.config['training']['save_freq'] == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pth')
        
        print("Training completed!")
        
        # Test on test set
        try:
            self.test()
        except Exception as e:
            print(f"⚠️ Testing failed: {e}")


if __name__ == "__main__":
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create trainer
    trainer = FaceAgingTrainer(config)
    
    # Train
    trainer.train()
