!pip install opencv-python==4.9.0.80 numpy==1.26.4 scikit-learn==1.4.0 matplotlib==3.8.3

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import List, Tuple, Dict, Optional
import random
from IPython.display import FileLink, display
import pandas as pd
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

SUPPORTED_EXTS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
INPUT_DIR = "/kaggle/input/pokemon-images-and-types/images"
OUTPUT_DIR = "/kaggle/working/processed_pokemon"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# DIGITAL IMAGE PROCESSING PROJECT - WITH CSV FEATURE SAVING
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import List, Tuple, Dict, Optional
import random
from IPython.display import FileLink, display
import pandas as pd
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ========================
# 1. CONSTANTS AND SETUP
# ========================
SUPPORTED_EXTS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
INPUT_DIR = "/kaggle/input/pokemon-images-and-types/images"
OUTPUT_DIR = "/kaggle/working/processed_pokemon"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========================
# 2. CORE FUNCTIONS (UNCHANGED)
# ========================
def load_images(folder_path: str, max_images: int = 50) -> Tuple[List[np.ndarray], List[str]]:
    """Load images with robust error handling and transparency support"""
    images = []
    filenames = []
    
    print(f"🔍 Loading images from {folder_path}...")
    for filename in tqdm(sorted(os.listdir(folder_path))[:max_images], desc="Loading"):
        try:
            if not filename.lower().endswith(SUPPORTED_EXTS):
                continue
                
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            
            if img is None:
                raise ValueError("File could not be read")
                
            # Handle transparency
            if img.shape[-1] == 4:
                alpha = img[:,:,3:4]/255.0
                img = img[:,:,:3]*alpha + 255*(1-alpha)
                img = img.astype(np.uint8)
                
            images.append(img)
            filenames.append(filename)
            
        except Exception as e:
            print(f"⚠️ Error loading {filename}: {str(e)}")
    
    print(f"✅ Successfully loaded {len(images)} images")
    return images, filenames

def process_image(img: np.ndarray) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """Complete image processing pipeline"""
    results = {}
    
    # Processing steps
    results['original'] = img.copy()
    results['adjusted'] = cv2.convertScaleAbs(img, alpha=1.2, beta=20)  # Brightness/contrast
    results['blurred'] = cv2.GaussianBlur(results['adjusted'], (5,5), 0)  # Gaussian blur
    gray = cv2.cvtColor(results['blurred'], cv2.COLOR_BGR2GRAY)
    results['edges'] = cv2.Canny(gray, 50, 150)  # Edge detection
    results['denoised'] = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)  # Denoising
    
    # Feature extraction
    features = {
        'color_hist': cv2.calcHist([img], [0,1,2], None, [32,32,32], [0,256,0,256,0,256]),
        'edge_density': float(np.mean(results['edges'] > 0)),
        'aspect_ratio': img.shape[1]/img.shape[0]
    }
    
    return results, features

def save_outputs(results: Dict[str, np.ndarray], base_filename: str):
    """Save processed images with organized structure"""
    try:
        # Create subdirectories
        for subdir in ['denoised', 'edges', 'collages']:
            os.makedirs(f"{OUTPUT_DIR}/{subdir}", exist_ok=True)
        
        # Save individual images
        cv2.imwrite(f"{OUTPUT_DIR}/denoised/denoised_{base_filename}.png", results['denoised'])
        cv2.imwrite(f"{OUTPUT_DIR}/edges/edges_{base_filename}.png", results['edges'])
        
        # Create and save collage
        plt.figure(figsize=(12,3))
        titles = ['Original', 'Adjusted', 'Blurred', 'Edges', 'Denoised']
        for i, (key, img) in enumerate(results.items()):
            plt.subplot(1,5,i+1)
            if key == 'edges':
                plt.imshow(img, cmap='gray')
            else:
                plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.title(titles[i])
            plt.axis('off')
        plt.suptitle(f"Processing: {base_filename.split('.')[0]}")
        plt.savefig(f"{OUTPUT_DIR}/collages/collage_{base_filename}.png", bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"⚠️ Failed to save {base_filename}: {str(e)}")

# ========================
# 3. VISUALIZATION HELPERS (UNCHANGED)
# ========================
def show_sample_results(n_samples=3):
    """Display random processed images"""
    samples = random.sample(os.listdir(f"{OUTPUT_DIR}/edges"), min(n_samples, 5))
    plt.figure(figsize=(12,4))
    for i, filename in enumerate(samples):
        img = cv2.imread(f"{OUTPUT_DIR}/edges/{filename}", cv2.IMREAD_GRAYSCALE)
        plt.subplot(1,n_samples,i+1)
        plt.imshow(img, cmap='gray')
        plt.title(filename.split('_')[1])
        plt.axis('off')
    plt.show()

# ========================
# 4. CLASSIFICATION PIPELINE (UNCHANGED)
# ========================
def run_classification(features: List[Dict]):
    """Optional classification task"""
    if len(features) < 10:
        print("⚠️ Not enough samples for classification (need ≥10)")
        return
    
    # Prepare features
    X = []
    y = []
    for feat in features:
        try:
            X.append(np.concatenate([
                feat['color_hist'].flatten(),
                [feat['edge_density']],
                [feat['aspect_ratio']]
            ]))
            y.append(feat.get('name', 'unknown'))
        except:
            continue
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train and evaluate
    clf = RandomForestClassifier(n_estimators=50, random_state=42)
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)

# ========================
# 5. FEATURE SAVING (NEW)
# ========================
def save_features_to_csv(features: List[Dict], csv_name: str = "pokemon_features.csv"):
    """Save extracted features to CSV file"""
    try:
        # Create DataFrame
        features_df = pd.DataFrame(features)
        
        # Flatten color histograms for CSV storage
        features_df['color_hist'] = features_df['color_hist'].apply(
            lambda x: x.flatten().tolist()
        )
        
        # Save to CSV
        csv_path = f"{OUTPUT_DIR}/{csv_name}"
        features_df.to_csv(csv_path, index=False)
        print(f"✅ Features saved to {csv_path}")
        
        # Show sample of saved data
        print("\nSample features from CSV:")
        print(features_df.head(3)[['name', 'edge_density', 'aspect_ratio']])
        
        return csv_path
    except Exception as e:
        print(f"⚠️ Failed to save features: {str(e)}")
        return None

# ========================
# 6. SUBMISSION GENERATION (UPDATED)
# ========================
def create_submission():
    """Package results for submission including features CSV"""
    import shutil
    
    # Create clean submission folder
    submission_dir = "/kaggle/working/submission"
    if os.path.exists(submission_dir):
        shutil.rmtree(submission_dir)
    os.makedirs(submission_dir)
    
    # Copy essential files
    shutil.copytree(OUTPUT_DIR, f"{submission_dir}/processed_images")
    
    # Create ZIP
    shutil.make_archive("/kaggle/working/pokemon_processing", 'zip', submission_dir)
    print("\n✅ Submission package created: pokemon_processing.zip")
    print("Includes:")
    print(f"- {len(os.listdir(f'{submission_dir}/processed_images/denoised'))} denoised images")
    print(f"- {len(os.listdir(f'{submission_dir}/processed_images/edges'))} edge maps")
    print(f"- Features CSV with {len(pd.read_csv(f'{submission_dir}/processed_images/pokemon_features.csv'))} entries")
    
    display(FileLink('/kaggle/working/pokemon_processing.zip'))

# ========================
# 7. MAIN EXECUTION (UPDATED)
# ========================
def main():
    """Complete processing pipeline with feature saving"""
    # Load images
    images, filenames = load_images(INPUT_DIR, max_images=50)
    if not images:
        raise RuntimeError("No valid images found!")
    
    # Process images
    all_features = []
    for img, filename in tqdm(zip(images, filenames), desc="Processing", total=len(images)):
        try:
            results, features = process_image(img)
            features['name'] = os.path.splitext(filename)[0]  # Add Pokémon name
            save_outputs(results, filename)
            all_features.append(features)
        except Exception as e:
            print(f"⚠️ Error processing {filename}: {str(e)}")
    
    # Save features to CSV
    save_features_to_csv(all_features)
    
    # Show results
    print(f"\n🎉 Successfully processed {len(all_features)} images")
    show_sample_results()
    
    # Optional classification
    run_classification(all_features)
    
    # Create submission package
    create_submission()

if __name__ == "__main__":
    main()
