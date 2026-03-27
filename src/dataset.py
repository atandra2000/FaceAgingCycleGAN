"""Dataset handling for IMDB-WIKI face aging dataset. Includes data loading, preprocessing, and augmentation"""

import os
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np
from scipy.io import loadmat
from datetime import datetime
from collections import Counter
from sklearn.model_selection import train_test_split

def find_image_path(image_dir, filename):
    """Try multiple path patterns to find the image"""
    # Pattern 1: subfolder by first 2 chars (standard IMDB-WIKI)
    path1 = os.path.join(image_dir, filename[:2], filename)
    if os.path.exists(path1):
        return path1
    # Pattern 2: direct in image_dir
    path2 = os.path.join(image_dir, filename)
    if os.path.exists(path2):
        return path2
    # Pattern 3: search subdirectories
    for root, dirs, files in os.walk(image_dir):
        if filename in files:
            return os.path.join(root, filename)
    return None

class IMDBWIKIDataset(Dataset):
    """IMDB-WIKI dataset for face aging. Handles both portions of the dataset"""
    def __init__(self, data_root,
                 dataset_type='imdb', # 'imdb', 'wiki', or 'combined'
                 split='train', image_size=512, age_groups=None, transform=None,
                 min_age=0, max_age=100, samples=None, cache_size=1000):
        
        self._cache = {}  # Dict for preloaded numpy arrays
        self._cache_order = []  # Track insertion order for LRU eviction
        self.cache_size = cache_size  # Max number of cached items
        self.data_root = data_root
        self.dataset_type = dataset_type
        self.split = split
        self.image_size = image_size
        self.transform = transform
        self.min_age = min_age
        self.max_age = max_age

        if age_groups is None:
            self.age_groups = {'young': [0, 30], 'old': [50, 100], 'middle': [31, 49]}
        else:
            self.age_groups = age_groups
        # Use provided samples if given (for combined datasets), else load
        if samples is not None:
            self.samples = samples
        else:
            self.samples = self._load_dataset()
        print(f"Loaded {len(self.samples)} samples from {dataset_type} {split} split")
        self._print_age_distribution()

    def _load_dataset(self):
        """Load dataset metadata from .mat files, robust to indexing and path variations."""
        if self.dataset_type == 'imdb':
            mat_file = os.path.join(self.data_root, 'imdb_crop_clean', 'imdb_crop', 'imdb.mat')
            image_dir = os.path.join(self.data_root, 'imdb_crop_clean', 'imdb_crop')
        elif self.dataset_type == 'wiki':
            mat_file = os.path.join(self.data_root, 'wiki_crop_clean', 'wiki_crop', 'wiki.mat')
            image_dir = os.path.join(self.data_root, 'wiki_crop_clean', 'wiki_crop')
        else:
            raise ValueError(f"Unknown dataset_type: {self.dataset_type}")
        if not os.path.exists(mat_file):
            raise FileNotFoundError(f"Dataset file not found: {mat_file}")
        # Load .mat file
        mat_data = loadmat(mat_file)
        # Extract metadata
        if self.dataset_type == 'imdb':
            data = mat_data['imdb'][0, 0]
        else:
            data = mat_data['wiki'][0, 0]
        full_paths = data['full_path'][0]
        dobs = data['dob']
        photo_takens = data['photo_taken']
        face_locations = data['face_location']
        genders = data['gender']

        samples = []
        def safe_extract(arr, idx, use_nested=False):
            """Safely extract value from MATLAB array"""
            try:
                if use_nested and arr.ndim >= 2:
                    val = arr[0, idx]
                else:
                    val = arr.flat[idx] if arr.size > idx else arr[idx]
                return val
            except (IndexError, ValueError):
                try:
                    return arr.flat[idx]
                except Exception:
                    return np.nan
        use_nested = (self.dataset_type == 'imdb')

        for i in range(len(full_paths)):
            try:
                dob = safe_extract(dobs, i, use_nested)
                photo_taken = safe_extract(photo_takens, i, use_nested)

                if np.isnan(dob) or np.isnan(photo_taken) or dob <= 0:
                    continue
                try:
                    birth_year = datetime.fromordinal(int(dob) - 366).year
                except Exception:
                    continue
                try:
                    age = int(photo_taken - birth_year)
                    if age < 0 or age > 120:  # Sanity check
                        continue
                except (ValueError, OverflowError):
                    continue
                if age < self.min_age or age > self.max_age or age < 0:
                    continue
                filename = full_paths[i][0]
                img_path = find_image_path(image_dir, filename)
                if img_path is None:
                    continue
                # Get face location
                try:
                    face_loc_raw = safe_extract(face_locations, i, False)
                    if isinstance(face_loc_raw, np.ndarray) and face_loc_raw.size >= 4:
                        face_loc = face_loc_raw.flatten()[:4].tolist()
                    else:
                        face_loc = None
                except Exception:
                    face_loc = None
                # Get gender
                try:
                    gender_raw = safe_extract(genders, i, use_nested)
                    gender = int(gender_raw) if not np.isnan(gender_raw) else -1
                except Exception:
                    gender = -1
                age_group = self._get_age_group(age)
                sample = {
                    'image_path': img_path,
                    'age': age,
                    'age_group': age_group,
                    'gender': gender,
                    'face_location': face_loc,
                    'dataset': self.dataset_type
                }
                samples.append(sample)
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                continue
        print(f"Successfully loaded {len(samples)} samples from {len(full_paths)} total entries")

        # Split dataset
        samples = self._split_dataset(samples)
        return samples

    def _get_age_group(self, age):
        """Determine age group (young/old/middle) based on age"""
        for group, (low, high) in self.age_groups.items():
            if low <= age <= high:
                return group
        return 'unknown'  # Fallback for ages outside defined groups

    def _split_dataset(self, samples):
        """Stratified split dataset into train/val/test by age group"""
        if not samples:
            return []
        labels = [s['age_group'] for s in samples]
        # Train-test split (90% train+val, 10% test)
        train_val, test = train_test_split(samples, test_size=0.1, stratify=labels, random_state=42)
        # Train-val split (80% train, 20% val of train_val)
        labels_train_val = [s['age_group'] for s in train_val]
        train, val = train_test_split(train_val, test_size=0.2, stratify=labels_train_val, random_state=42)
        if self.split == 'train':
            return train
        elif self.split == 'val':
            return val
        else:
            return test

    def _print_age_distribution(self):
        """Print age distribution statistics including gender breakdown"""
        if not self.samples:
            print("No samples to display distribution.")
            return
        ages = [sample['age'] for sample in self.samples]
        age_groups = [sample['age_group'] for sample in self.samples]
        genders = [sample['gender'] for sample in self.samples]

        print(f"Age range: {min(ages)}-{max(ages)}")
        print(f"Mean age: {np.mean(ages):.1f}")

        group_counts = Counter(age_groups)
        gender_counts = Counter(genders)

        print(f"Age groups: {dict(group_counts)}")
        print(f"Gender distribution: Male={gender_counts.get(1, 0)}, Female={gender_counts.get(0, 0)}, Unknown={gender_counts.get(-1, 0)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """Get a single sample with proper caching and augmentation"""
        sample = self.samples[idx]
        age = sample['age']
    
        # Create cache key
        cache_key = f"{self.dataset_type}_{sample['image_path']}"
        pil_image = None
    
        # Try to get from cache
        if cache_key in self._cache:
            image_np, cached_age = self._cache[cache_key]
            assert age == cached_age, "Age mismatch in cache"
            pil_image = Image.fromarray(image_np)
        else:
            # Load and preprocess image
            try:
                pil_image = Image.open(sample['image_path']).convert('RGB')
            except Exception as e:
                print(f"Error loading image {sample['image_path']}: {e}")
                return self.__getitem__((idx + 1) % len(self.samples))
        
            # Crop face if location available
            if sample['face_location'] is not None:
                try:
                    face_loc = sample['face_location']
                    pil_image = pil_image.crop((
                        face_loc[0], face_loc[1],
                        face_loc[0] + face_loc[2],
                        face_loc[1] + face_loc[3]
                    ))
                except Exception as e:
                    print(f"Error cropping face: {e}")
        
            # Resize to target size
            pil_image = pil_image.resize((self.image_size, self.image_size), Image.BILINEAR)
        
            # Convert to numpy array for caching
            image_np = np.array(pil_image, dtype=np.uint8)
        
            # Cache management with LRU eviction
            if len(self._cache) >= self.cache_size:
                # Remove oldest entry (FIFO for simplicity)
                oldest_key = self._cache_order.pop(0)
                del self._cache[oldest_key]
            
            self._cache[cache_key] = (image_np.copy(), age)
            self._cache_order.append(cache_key)  # Track insertion order
    
        # Apply augmentations - handle both albumentations and torchvision
        image_tensor = None
        if self.transform is not None:
            try:
                # Check if it's albumentations (has ToTensorV2)
                is_albumentations = hasattr(self.transform, 'transforms') and any(
                    'ToTensorV2' in str(type(t)) for t in self.transform.transforms
                )
            
                if is_albumentations:
                    # Albumentations expects named argument
                    image_np = np.array(pil_image, dtype=np.uint8)
                    augmented = self.transform(image=image_np)
                    image_tensor = augmented['image']
                else:
                    # Torchvision expects PIL image
                    image_tensor = self.transform(pil_image)
            except Exception as e:
                print(f"Transform error: {e}, using manual conversion")
                # Manual fallback
                image_np = np.array(pil_image, dtype=np.uint8)
                image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
                image_tensor = (image_tensor - 0.5) / 0.5
        else:
            # No transform: manual conversion
            image_np = np.array(pil_image, dtype=np.uint8)
            image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
            image_tensor = (image_tensor - 0.5) / 0.5
    
        return {
            'image': image_tensor,
            'age': torch.tensor(age, dtype=torch.long),
            'path': sample['image_path']
        }


class FaceAgingDataModule:
    """Data module for face aging that handles train/val/test splits"""
    def __init__(self, config): 
        self.config = config
        self.data_root = self.config['data']['data_root']
        self.batch_size = self.config['training']['batch_size']
        self.image_size = self.config['data']['image_size']
        self.num_workers = self.config['data'].get('num_workers', 4)
        self.prefetch_factor = self.config['data'].get('prefetch_factor', 2)
        self.age_groups = self.config['data'].get('age_groups')
        self.balance_classes = self.config['data'].get('balance_classes', False)
        self.cache_size = self.config['data'].get('cache_size', 1000)
        self.seed = self.config['training']['seed']

        # Define augmentations (using albumentations if specified)
        if self.config.get('data', {}).get('use_albumentations', True):
            aug_prob = self.config.get('data', {}).get('albumentations_probability', 0.8)
            
            self.train_transforms = A.Compose([
                A.Resize(self.image_size, self.image_size),
                A.OneOf([
                    A.CLAHE(clip_limit=2.0, p=1.0),
                    A.HorizontalFlip(p=1.0),
                    A.ElasticTransform(alpha=1, sigma=50, p=1.0),
                    A.RandomBrightnessContrast(p=1.0),
                ], p=aug_prob),
                A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ToTensorV2()
            ])
            self.val_transforms = A.Compose([
                A.Resize(self.image_size, self.image_size),
                A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ToTensorV2()
            ])
        else:
            # Fallback to torchvision
            self.train_transforms = transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            self.val_transforms = transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

    def get_datasets(self):
        """Get train, validation, and test datasets"""
        # Load IMDB
        imdb_train = IMDBWIKIDataset(self.data_root, 'imdb', 'train', self.image_size, self.age_groups, self.train_transforms)
        imdb_val = IMDBWIKIDataset(self.data_root, 'imdb', 'val', self.image_size, self.age_groups, self.val_transforms)
        imdb_test = IMDBWIKIDataset(self.data_root, 'imdb', 'test', self.image_size, self.age_groups, self.val_transforms)

        # Load WIKI if available
        try:
            wiki_train = IMDBWIKIDataset(self.data_root, 'wiki', 'train', self.image_size, self.age_groups, self.train_transforms)
            wiki_val = IMDBWIKIDataset(self.data_root, 'wiki', 'val', self.image_size, self.age_groups, self.val_transforms)
            wiki_test = IMDBWIKIDataset(self.data_root, 'wiki', 'test', self.image_size, self.age_groups, self.val_transforms)

            # Combine samples
            train_samples = imdb_train.samples + wiki_train.samples
            val_samples = imdb_val.samples + wiki_val.samples
            test_samples = imdb_test.samples + wiki_test.samples

            # Create combined datasets using pre-loaded samples (avoids re-loading)
            combined_train = IMDBWIKIDataset(self.data_root, 'combined', 'train', self.image_size, self.age_groups, self.train_transforms, samples=train_samples)
            combined_val = IMDBWIKIDataset(self.data_root, 'combined', 'val', self.image_size, self.age_groups, self.val_transforms, samples=val_samples)
            combined_test = IMDBWIKIDataset(self.data_root, 'combined', 'test', self.image_size, self.age_groups, self.val_transforms, samples=test_samples)

            return combined_train, combined_val, combined_test

        except Exception as e:
            print(f"Wiki dataset not found or error: {e}, using IMDB only")
            return imdb_train, imdb_val, imdb_test

    def _create_sampler(self, dataset):
        """Create WeightedRandomSampler for class balancing if enabled"""
        if not self.balance_classes:
            return None
        age_groups = [sample['age_group'] for sample in dataset.samples]
        class_counts = Counter(age_groups)
        if len(class_counts) < 2:
            return None  # No balancing needed if only one class
        weights = [1.0 / class_counts[ag] for ag in age_groups]
        return WeightedRandomSampler(weights, len(weights), replacement=True)

    def get_domain_dataloaders(self):
        """Get separate DataLoaders for young and old domains"""
        train_dataset, val_dataset, test_dataset = self.get_datasets()

        young_max = self.config['data']['young_max_age']
        old_min = self.config['data']['old_min_age']

        # Filter for young
        young_train_samples = [s for s in train_dataset.samples if s['age'] <= young_max]
        young_val_samples = [s for s in val_dataset.samples if s['age'] <= young_max]
        young_test_samples = [s for s in test_dataset.samples if s['age'] <= young_max]

        young_train = IMDBWIKIDataset(self.data_root, 'young', 'train', self.image_size, self.age_groups, self.train_transforms, samples=young_train_samples)
        young_val = IMDBWIKIDataset(self.data_root, 'young', 'val', self.image_size, self.age_groups, self.val_transforms, samples=young_val_samples)
        young_test = IMDBWIKIDataset(self.data_root, 'young', 'test', self.image_size, self.age_groups, self.val_transforms, samples=young_test_samples)

        # Filter for old
        old_train_samples = [s for s in train_dataset.samples if s['age'] >= old_min]
        old_val_samples = [s for s in val_dataset.samples if s['age'] >= old_min]
        old_test_samples = [s for s in test_dataset.samples if s['age'] >= old_min]

        old_train = IMDBWIKIDataset(self.data_root, 'old', 'train', self.image_size, self.age_groups, self.train_transforms, samples=old_train_samples)
        old_val = IMDBWIKIDataset(self.data_root, 'old', 'val', self.image_size, self.age_groups, self.val_transforms, samples=old_val_samples)
        old_test = IMDBWIKIDataset(self.data_root, 'old', 'test', self.image_size, self.age_groups, self.val_transforms, samples=old_test_samples)

        # Create loaders with balancing if enabled
        young_train_sampler = self._create_sampler(young_train) if self.balance_classes else None
        old_train_sampler = self._create_sampler(old_train) if self.balance_classes else None

        young_train_loader = DataLoader(
            young_train,
            batch_size=self.batch_size,
            shuffle=(young_train_sampler is None),
            sampler=young_train_sampler,
            num_workers=self.num_workers,
            worker_init_fn=lambda worker_id: np.random.seed(self.seed + worker_id),
            persistent_workers=True,
            prefetch_factor=self.prefetch_factor,
            pin_memory=True,
            drop_last=True
        )
        old_train_loader = DataLoader(
            old_train,
            batch_size=self.batch_size,
            shuffle=(old_train_sampler is None),
            sampler=old_train_sampler,
            num_workers=self.num_workers,
            worker_init_fn=lambda worker_id: np.random.seed(self.seed + worker_id),
            persistent_workers=True,
            prefetch_factor=self.prefetch_factor,
            pin_memory=True,
            drop_last=True
        )
        young_val_loader = DataLoader(
            young_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True
        )
        old_val_loader = DataLoader(
            old_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            worker_init_fn=lambda worker_id: np.random.seed(self.seed + worker_id),
            persistent_workers=True,
            pin_memory=True
        )
        young_test_loader = DataLoader(
            young_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            worker_init_fn=lambda worker_id: np.random.seed(self.seed + worker_id),
            persistent_workers=True,
            pin_memory=True
        )
        old_test_loader = DataLoader(
            old_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            worker_init_fn=lambda worker_id: np.random.seed(self.seed + worker_id),
            persistent_workers=True,
            pin_memory=True
        )
        return (young_train_loader, old_train_loader), (young_val_loader, old_val_loader), (young_test_loader, old_test_loader)
    