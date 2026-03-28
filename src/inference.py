"""
Age-Aware CycleGAN Inference Script
Customized for your trained model (31 epochs)
"""

import torch
from PIL import Image
import numpy as np
import argparse
import os
from pathlib import Path
import torchvision.transforms as transforms
import yaml

# Import your actual model classes
from cyclegan import FaceAgingCycleGAN


class AgeTransformer:
    """Easy-to-use interface for age transformation inference"""
    
    def __init__(self, checkpoint_path, config_path='config.yaml', device='cuda'):
        """
        Initialize the age transformer
        
        Args:
            checkpoint_path: Path to trained checkpoint (.pth file)
            config_path: Path to config.yaml
            device: 'cuda' or 'cpu'
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Build model
        self.model = self._build_model()
        
        # Load checkpoint
        self._load_checkpoint(checkpoint_path)
        
        # Set to eval mode
        self.model.eval()
        
        # Image transforms (matching training)
        self.transform = transforms.Compose([
            transforms.Resize((self.config['data']['image_size'], 
                             self.config['data']['image_size'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        self.inverse_transform = transforms.Compose([
            transforms.Normalize(mean=[-1, -1, -1], std=[2, 2, 2]),
            transforms.ToPILImage()
        ])
    
    def _build_model(self):
        """Build the FaceAgingCycleGAN model from config"""
        model = FaceAgingCycleGAN(
            input_nc=self.config['model']['input_nc'],
            output_nc=self.config['model']['output_nc'],
            ngf=self.config['model']['ngf'],
            ndf=self.config['model']['ndf'],
            n_residual_blocks=self.config['model']['n_residual_blocks'],
            num_ages=self.config['model']['num_ages']
        )
        return model.to(self.device)
    
    def _load_checkpoint(self, checkpoint_path):
        """Load trained weights from checkpoint"""
        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
    
        # Your checkpoint format: single model_state_dict with all components
        self.model.load_state_dict(checkpoint['model_state_dict'])
    
        epoch = checkpoint.get('epoch', 'unknown')
        best_val_loss = checkpoint.get('best_val_loss', 'N/A')
    
        print(f"✓ Loaded checkpoint from epoch {epoch}")
        print(f"  Best validation loss: {best_val_loss}")
    
        # Show file size
        import os
        size_mb = os.path.getsize(checkpoint_path) / (1024*1024)
        print(f"  Checkpoint size: {size_mb:.1f} MB")

    def load_image(self, image_path):
        """Load and preprocess an image"""
        img = Image.open(image_path).convert('RGB')
        original_size = img.size
        transformed = self.transform(img).unsqueeze(0).to(self.device)
        return transformed, original_size
    
    def save_image_tensor(self, tensor, output_path, original_size=None):
        """Save tensor as image, optionally resize to original"""
        img = self.inverse_transform(tensor.cpu().squeeze(0))
        
        # Optionally resize back to original dimensions
        if original_size:
            img = img.resize(original_size, Image.LANCZOS)
        
        img.save(output_path, quality=95)
        return img
    
    @torch.no_grad()
    def age_face(self, image_path, target_age, direction='young_to_old', 
                 output_path=None, keep_original_size=True):
        """
        Transform face to target age
        
        Args:
            image_path: Path to input image
            target_age: Target age (0-100)
            direction: 'young_to_old' or 'old_to_young'
            output_path: Where to save result (optional)
            keep_original_size: Resize output to match input size
        
        Returns:
            PIL Image of transformed face
        """
        # Validate age
        if not 0 <= target_age <= 100:
            raise ValueError(f"Target age must be 0-100, got {target_age}")
        
        # Load image
        img_tensor, original_size = self.load_image(image_path)
        
        # Prepare age tensor (as integer, NOT one-hot)
        age_tensor = torch.tensor([target_age], dtype=torch.long).to(self.device)
        
        # Generate transformed image
        if direction == 'young_to_old':
            output = self.model.G_Y2O(img_tensor, age_tensor)
        elif direction == 'old_to_young':
            output = self.model.G_O2Y(img_tensor, age_tensor)
        else:
            raise ValueError(f"Unknown direction: {direction}. Use 'young_to_old' or 'old_to_young'")
        
        # Save or return
        if output_path:
            result_img = self.save_image_tensor(
                output, output_path, 
                original_size if keep_original_size else None
            )
            print(f"✓ Saved to: {output_path}")
        else:
            result_img = self.inverse_transform(output.cpu().squeeze(0))
            if keep_original_size:
                result_img = result_img.resize(original_size, Image.LANCZOS)
        
        return result_img
    
    @torch.no_grad()
    def age_progression(self, image_path, start_age, end_age, num_steps=7, 
                        output_dir='progression', direction='auto'):
        """
        Generate age progression sequence
        
        Args:
            image_path: Path to input image
            start_age: Starting age (0-100)
            end_age: Ending age (0-100)
            num_steps: Number of steps in progression
            output_dir: Directory to save sequence
            direction: 'auto', 'young_to_old', or 'old_to_young'
        
        Returns:
            List of PIL Images
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Auto-detect direction
        if direction == 'auto':
            direction = 'young_to_old' if end_age > start_age else 'old_to_young'
        
        # Generate age range
        ages = np.linspace(start_age, end_age, num_steps, dtype=int)
        
        # Save original image too
        original_img = Image.open(image_path)
        original_path = os.path.join(output_dir, 'age_original.jpg')
        original_img.save(original_path)
        print(f"Saved original: {original_path}")
        
        results = [original_img]
        
        for i, age in enumerate(ages):
            output_path = os.path.join(output_dir, f'age_{age:03d}.jpg')
            img = self.age_face(image_path, int(age), direction, output_path)
            results.append(img)
            print(f"Generated: {age} years ({i+1}/{num_steps})")
        
        # Create comparison grid
        self._create_comparison_grid(results, ages, output_dir)
        
        return results
    
    def _create_comparison_grid(self, images, ages, output_dir):
        """Create a comparison grid of all age progressions"""
        try:
            import matplotlib.pyplot as plt
            
            n_images = len(images)
            fig, axes = plt.subplots(1, n_images, figsize=(3*n_images, 3))
            
            if n_images == 1:
                axes = [axes]
            
            axes[0].imshow(images[0])
            axes[0].set_title('Original')
            axes[0].axis('off')
            
            for i, (img, age) in enumerate(zip(images[1:], ages), 1):
                axes[i].imshow(img)
                axes[i].set_title(f'Age {age}')
                axes[i].axis('off')
            
            plt.tight_layout()
            grid_path = os.path.join(output_dir, 'comparison_grid.jpg')
            plt.savefig(grid_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"✓ Saved comparison grid: {grid_path}")
        except ImportError:
            print("Note: Install matplotlib for comparison grid generation")
    
    @torch.no_grad()
    def batch_transform(self, input_dir, target_age, output_dir, direction='young_to_old'):
        """
        Transform all images in a directory
        
        Args:
            input_dir: Directory with input images
            target_age: Target age for all images
            output_dir: Directory to save results
            direction: Transformation direction
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Supported extensions
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
        
        # Find all images
        image_files = []
        for ext in extensions:
            image_files.extend(Path(input_dir).glob(f'*{ext}'))
            image_files.extend(Path(input_dir).glob(f'*{ext.upper()}'))
        
        if not image_files:
            print(f"No images found in {input_dir}")
            return
        
        print(f"Found {len(image_files)} images to process")
        
        for i, img_path in enumerate(image_files, 1):
            output_path = os.path.join(
                output_dir, 
                f'{img_path.stem}_age{target_age}{img_path.suffix}'
            )
            try:
                self.age_face(str(img_path), target_age, direction, output_path)
                print(f"[{i}/{len(image_files)}] ✓ {img_path.name}")
            except Exception as e:
                print(f"[{i}/{len(image_files)}] ✗ {img_path.name}: {e}")
        
        print(f"\n✓ Batch processing complete! Results in: {output_dir}")
    
    @torch.no_grad()
    def estimate_age(self, image_path):
        """
        Estimate age of face in image using model's age estimator
        
        Args:
            image_path: Path to image
            
        Returns:
            Estimated age (float)
        """
        img_tensor, _ = self.load_image(image_path)
        
        # Use model's built-in age estimator
        pred_age, saliency = self.model.estimate_age(img_tensor)
        
        estimated_age = pred_age.item()
        print(f"Estimated age: {estimated_age:.1f} years")
        
        return estimated_age


def main():
    parser = argparse.ArgumentParser(
        description='Age-Aware CycleGAN Inference (31-epoch model)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint file (e.g., checkpoint_epoch31.pth)')
    parser.add_argument('--input', type=str, required=True,
                       help='Input image or directory')
    
    # Optional arguments
    parser.add_argument('--output', type=str, default='output',
                       help='Output path/directory')
    parser.add_argument('--target_age', type=int, default=70,
                       help='Target age (0-100)')
    parser.add_argument('--direction', type=str, default='young_to_old',
                       choices=['young_to_old', 'old_to_young', 'auto'],
                       help='Transformation direction')
    parser.add_argument('--mode', type=str, default='single',
                       choices=['single', 'batch', 'progression', 'estimate'],
                       help='Processing mode')
    
    # Progression mode arguments
    parser.add_argument('--start_age', type=int, default=20,
                       help='Start age for progression mode')
    parser.add_argument('--end_age', type=int, default=80,
                       help='End age for progression mode')
    parser.add_argument('--num_steps', type=int, default=7,
                       help='Number of steps in progression')
    
    # System arguments
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use')
    parser.add_argument('--no_resize', action='store_true',
                       help='Keep output at 256x256 (don\'t resize to original)')
    
    args = parser.parse_args()
    
    # Banner
    print("=" * 70)
    print("  Age-Aware CycleGAN Inference")
    print("  Model: 31 epochs trained, 171 MB checkpoint")
    print("=" * 70)
    print()
    
    # Initialize transformer
    print("Initializing model...")
    transformer = AgeTransformer(args.checkpoint, args.config, args.device)
    print()
    
    # Process based on mode
    if args.mode == 'single':
        print("=== Single Image Mode ===")
        print(f"Input: {args.input}")
        print(f"Target age: {args.target_age}")
        print(f"Direction: {args.direction}")
        print()
        
        transformer.age_face(
            args.input, args.target_age, args.direction, 
            args.output, keep_original_size=not args.no_resize
        )
        print(f"\n✓ Done! Output saved to: {args.output}")
    
    elif args.mode == 'batch':
        print("=== Batch Processing Mode ===")
        print(f"Input directory: {args.input}")
        print(f"Target age: {args.target_age}")
        print(f"Direction: {args.direction}")
        print()
        
        transformer.batch_transform(args.input, args.target_age, args.output, args.direction)
    
    elif args.mode == 'progression':
        print("=== Age Progression Mode ===")
        print(f"Input: {args.input}")
        print(f"Age range: {args.start_age} → {args.end_age}")
        print(f"Steps: {args.num_steps}")
        print()
        
        transformer.age_progression(
            args.input, args.start_age, args.end_age,
            args.num_steps, args.output, args.direction
        )
        print(f"\n✓ Done! Progression saved to: {args.output}/")
    
    elif args.mode == 'estimate':
        print("=== Age Estimation Mode ===")
        print(f"Input: {args.input}")
        print()
        
        estimated_age = transformer.estimate_age(args.input)
        print(f"\n✓ Estimated age: {estimated_age:.1f} years")


if __name__ == '__main__':
    main()
