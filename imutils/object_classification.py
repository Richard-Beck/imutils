import os
import pickle
import numpy as np
import torch
import cv2
from torchvision import models, transforms
from sklearn.ensemble import RandomForestClassifier
from scipy.ndimage import find_objects

class ObjectClassifier:
    """
    Handles feature extraction and classification using RandomForestClassifier.
    - Uses a combined feature vector from a tight crop and a surrounding context crop.
    """
    # Version number for the feature format. Increment if features change.
    FEATURE_VERSION = 2

    def __init__(self, crop_size=224):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"ObjectClassifier using device: {self.device}")

        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone.fc = torch.nn.Identity()
        self.backbone.to(self.device)
        self.backbone.eval()

        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.is_trained = False
        self.crop_size = crop_size
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((crop_size, crop_size), antialias=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _get_combined_features(self, image, masks, object_ids):
        """Helper to extract, process, and combine features from both crops."""
        tight_crops, context_crops = self._extract_crops_and_preprocess(image, masks, object_ids)
        if not tight_crops:
            return None

        # Get embeddings for both crop types
        tight_embeddings = self._get_embeddings(torch.stack(tight_crops).to(self.device))
        context_embeddings = self._get_embeddings(torch.stack(context_crops).to(self.device))
        
        # Concatenate the two embeddings into one feature vector
        combined_features = np.concatenate([tight_embeddings, context_embeddings], axis=1)
        return combined_features

    def train_and_predict(self, all_images, all_masks, all_labels, predict_image, predict_masks) -> dict:
        """Trains on all labeled data and predicts on the current image."""
        print("Gathering training data from all images...")
        all_train_features, all_train_labels = [], []
        
        for i, (img, msk, lab) in enumerate(zip(all_images, all_masks, all_labels)):
            object_ids = [oid for oid, label_id in lab.items() if label_id != 0]
            if not object_ids: continue
            
            features = self._get_combined_features(img, msk, object_ids)
            if features is not None:
                all_train_features.append(features)
                all_train_labels.extend([lab[oid] for oid in object_ids])

        if not all_train_features:
            print("⚠️ Could not extract any features for training.")
            return {}

        X_train = np.concatenate(all_train_features)
        y_train = np.array(all_train_labels)
        
        if len(y_train) < 2 or len(np.unique(y_train)) < 2:
            print("⚠️ Insufficient data for training (less than 2 classes or samples).")
            return {}
        
        print(f"Training on {len(y_train)} annotations...")
        self.classifier.fit(X_train, y_train)
        self.is_trained = True

        return self.predict_only(predict_image, predict_masks)

    def predict_only(self, image: np.ndarray, masks: np.ndarray) -> dict:
        """Predicts labels for a single image, for use with the interactive editor."""
        if not self.is_trained: return {}
        
        all_ids_current = sorted([i for i in np.unique(masks) if i != 0])
        if not all_ids_current: return {}
        
        X_predict = self._get_combined_features(image, masks, all_ids_current)
        if X_predict is None: return {}
        
        predictions = self.classifier.predict(X_predict)
        return {int(obj_id): int(pred) for obj_id, pred in zip(all_ids_current, predictions)}

    def predict_with_probabilities(self, image: np.ndarray, masks: np.ndarray) -> dict:
        """Predicts labels and class probabilities, designed for batch processing."""
        if not self.is_trained: return {}

        all_ids_current = sorted([i for i in np.unique(masks) if i != 0])
        if not all_ids_current: return {}

        X_predict = self._get_combined_features(image, masks, all_ids_current)
        if X_predict is None: return {}
        
        predictions = self.classifier.predict(X_predict)
        probabilities = self.classifier.predict_proba(X_predict)
        
        return {
            int(obj_id): {'prediction': int(pred), 'probabilities': probs}
            for obj_id, pred, probs in zip(all_ids_current, predictions, probabilities)
        }

    def _extract_crops_and_preprocess(self, image: np.ndarray, masks: np.ndarray, object_ids: list) -> tuple:
        """Extracts a tight crop and an expanded context crop for each object."""
        tight_crop_tensors = []
        context_crop_tensors = []
        h, w, _ = image.shape
        locations = find_objects(masks)

        for obj_id in object_ids:
            if obj_id == 0 or obj_id > len(locations) or locations[obj_id - 1] is None: continue
            
            y_slice, x_slice = locations[obj_id - 1]

            # 1. Original tight crop (masked)
            img_crop_tight = image[y_slice, x_slice]
            mask_crop = (masks[y_slice, x_slice] == obj_id).astype(np.uint8)
            masked_img_crop = img_crop_tight * np.stack([mask_crop] * 3, axis=-1)
            tight_crop_tensors.append(self.transform(masked_img_crop))

            # 2. Expanded context crop (unmasked)
            cy, cx = (y_slice.start + y_slice.stop) // 2, (x_slice.start + x_slice.stop) // 2
            bh, bw = y_slice.stop - y_slice.start, x_slice.stop - x_slice.start
            
            y_start = max(0, cy - (bh * 3) // 2)
            y_end = min(h, cy + (bh * 3) // 2)
            x_start = max(0, cx - (bw * 3) // 2)
            x_end = min(w, cx + (bw * 3) // 2)
            
            img_crop_context = image[y_start:y_end, x_start:x_end]
            context_crop_tensors.append(self.transform(img_crop_context))

        if not tight_crop_tensors:
            return None, None
            
        return tight_crop_tensors, context_crop_tensors

    @torch.no_grad()
    def _get_embeddings(self, preprocessed_crops: torch.Tensor) -> np.ndarray:
        if preprocessed_crops.nelement() == 0: return np.array([])
        return self.backbone(preprocessed_crops).cpu().numpy()

    def save_state(self, path: str):
        """Saves the classifier state and feature version."""
        state = {
            'classifier': self.classifier,
            'is_trained': self.is_trained,
            'feature_version': self.FEATURE_VERSION
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        print(f"✅ Classifier state saved to {path}")

    def load_state(self, path: str):
        """Loads classifier state, checking for feature format compatibility."""
        print(f"✅ Loading classifier state from {path}")
        with open(path, 'rb') as f:
            state = pickle.load(f)
        
        loaded_version = state.get('feature_version', 1) # Default to 1 if no version is saved
        
        if loaded_version == self.FEATURE_VERSION and isinstance(state.get('classifier'), RandomForestClassifier):
            self.classifier = state['classifier']
            self.is_trained = state.get('is_trained', False)
            print(f"   Successfully loaded model version {loaded_version}.")
        else:
            print(f"⚠️ WARNING: Saved model (version {loaded_version}) is incompatible with current code (version {self.FEATURE_VERSION}).")
            print("   Starting with a new, untrained model.")
            self.is_trained = False
