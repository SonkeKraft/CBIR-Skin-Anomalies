# ============================================================
# IMPORT LIBRARIES
# ============================================================

import os                  # For file and folder operations
import cv2                 # OpenCV for image reading and processing
import pickle              # For saving/loading Python objects to disk
import random              # For optional shuffling of image file order
import shutil              # For copying result images into an output folder
import numpy as np         # For numerical operations on vectors and arrays
import matplotlib.pyplot as plt   # For displaying query and result images
import nmslib              # For fast nearest-neighbor search using HNSW indexing

from tqdm import tqdm      # Progress bar for loops
from skimage.feature import local_binary_pattern  # Texture descriptor (LBP)
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input
from tensorflow.keras.preprocessing import image


# ============================================================
# MAIN CBIR CLASS
# ============================================================

class SkinCBIRSystem:
    def __init__(
        self,
        feature_file="skin_subset_features.pkl",
        index_file="skin_index.nmslib",
        metadata_file="skin_metadata.pkl",
        cnn_weight=0.6,      #default 0.6
        color_weight=0.2,    #default 0.25
        texture_weight=0.2   #default 0.15
    ):
        # File used to store extracted features
        self.feature_file = feature_file

        # File used to store the NMSLIB index
        self.index_file = index_file

        # File used to store metadata such as image paths
        self.metadata_file = metadata_file

        # Weights used to control the influence of each descriptor
        self.cnn_weight = cnn_weight
        self.color_weight = color_weight
        self.texture_weight = texture_weight

        # List of dataset image file paths
        self.image_paths = []

        # List of extracted feature vectors for the dataset images
        self.features = []

        # Flag to know whether the HNSW index has already been built
        self.index_built = False

        # Load the CNN model used for semantic/deep feature extraction
        print("Loading EfficientNetB0 model...")
        self.cnn_model = EfficientNetB0(weights="imagenet", include_top=False, pooling="avg")
        print("Model loaded.")

        # NMSLIB index using HNSW graph structure and L2 (Euclidean) distance
        self.index = nmslib.init(
            method="hnsw",
            space="l2",
            data_type=nmslib.DataType.DENSE_VECTOR,
            dtype=nmslib.DistType.FLOAT  
            
        )

    def l2norm(self, x, eps=1e-8):
        # Convert input to float32 and normalize it to unit length
        # This makes different feature vectors comparable in scale
        x = np.asarray(x, dtype=np.float32)
        return x / (np.linalg.norm(x) + eps)

    def read_rgb(self, img_path):
        # Read image from disk using OpenCV
        img = cv2.imread(img_path)

        # If image cannot be read, return None
        if img is None:
            return None

        # OpenCV reads in BGR format, convert to RGB for consistent processing
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def extract_cnn_features(self, img_rgb):
        # Resize image to the input size expected by EfficientNetB0
        img_res = cv2.resize(img_rgb, (224, 224))

        # Convert image to array format compatible with Keras/TensorFlow
        x = image.img_to_array(img_res)

        # Add batch dimension: shape becomes (1, 224, 224, 3)
        x = np.expand_dims(x, axis=0)

        # Apply EfficientNet preprocessing
        x = preprocess_input(x)

        # Run image through the CNN and flatten the output into a 1D vector
        feat = self.cnn_model.predict(x, verbose=0).flatten().astype(np.float32)

        # Normalize the resulting CNN descriptor
        return self.l2norm(feat)

    def extract_color_features(self, img_rgb):
        # Resize image to keep descriptor computation consistent
        img_res = cv2.resize(img_rgb, (224, 224))

        # Convert image from RGB to HSV color space
        hsv = cv2.cvtColor(img_res, cv2.COLOR_RGB2HSV)

        # Compute histograms for each HSV channel separately
        # H = Hue, S = Saturation, V = Value/Brightness
        hist_h = cv2.calcHist([hsv], [0], None, [32], [0, 180]).flatten()
        hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256]).flatten()
        hist_v = cv2.calcHist([hsv], [2], None, [32], [0, 256]).flatten()

        # Concatenate the three histograms into a single color feature vector
        feat = np.concatenate([hist_h, hist_s, hist_v]).astype(np.float32)

        # Normalize the color descriptor
        return self.l2norm(feat)

    def extract_texture_features(self, img_rgb):
        # Resize image before texture extraction
        img_res = cv2.resize(img_rgb, (224, 224))

        # Convert to grayscale because LBP works on intensity values
        gray = cv2.cvtColor(img_res, cv2.COLOR_RGB2GRAY)

        # Define LBP parameters
        radius = 2
        n_points = 8 * radius

        # Compute Local Binary Pattern image
        lbp = local_binary_pattern(gray, n_points, radius, method="uniform")

        # Build a histogram of LBP codes
        n_bins = n_points + 2
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)

        # Normalize the texture descriptor
        return self.l2norm(hist.astype(np.float32))

    def extract_features(self, img_path):
        # Read image from disk
        img_rgb = self.read_rgb(img_path)

        # If image could not be loaded, return None
        if img_rgb is None:
            print(f"Warning: could not read image: {img_path}")
            return None

        # Extract the three descriptor types
        feat_cnn = self.extract_cnn_features(img_rgb)
        feat_color = self.extract_color_features(img_rgb)
        feat_texture = self.extract_texture_features(img_rgb)

        # Early fusion:
        # combine weighted CNN, color, and texture descriptors into one vector
        fused = np.concatenate([
            self.cnn_weight * feat_cnn,
            self.color_weight * feat_color,
            self.texture_weight * feat_texture
        ]).astype(np.float32)

        # Normalize the final fused descriptor
        return self.l2norm(fused)

    def load_image_subset(self, folder_path, max_images=100, shuffle=True):
        # Check whether dataset folder exists
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Dataset folder does not exist: {folder_path}")

        # Supported image file extensions
        valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

        # Build a list of all valid image file paths in the folder
        all_files = [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.lower().endswith(valid_exts)
        ]

        # Optionally shuffle the image list before selecting a subset
        if shuffle:
            random.shuffle(all_files)

        # Keep only the first max_images images
        self.image_paths = all_files[:max_images]
        print(f"Loaded {len(self.image_paths)} images from: {folder_path}")

        # Warn if no images were found
        if len(self.image_paths) == 0:
            print("No images found. Check the folder path and file extensions.")

    def build_feature_database(self):
        # Ensure images are loaded before extracting features
        if not self.image_paths:
            raise ValueError("No images loaded. Call load_image_subset() first.")

        # Reset the feature list and keep track of successfully processed images
        self.features = []
        valid_paths = []

        print("Extracting features for database images...")

        # Extract features for every image in the dataset
        for path in tqdm(self.image_paths):
            feat = self.extract_features(path)

            # Save only valid feature vectors
            if feat is not None:
                self.features.append(feat)
                valid_paths.append(path)

        # Update image path list so it only contains valid images
        self.image_paths = valid_paths

        # Save extracted features and image paths to disk
        with open(self.feature_file, "wb") as f:
            pickle.dump({
                "image_paths": self.image_paths,
                "features": self.features
            }, f)

        print(f"Feature database built for {len(self.image_paths)} images.")

    def build_index(self):
        # Ensure feature vectors exist before building the HNSW index
        if not self.features:
            raise ValueError("No features available. Call build_feature_database() first.")

        print("Building NMSLIB HNSW index...")

        # Convert list of features into a NumPy matrix
        data_matrix = np.asarray(self.features, dtype=np.float32)
            
        # Re-initialize the NMSLIB index
        self.index = nmslib.init(
            method="hnsw",
            space="l2",
            data_type=nmslib.DataType.DENSE_VECTOR,
            dtype=nmslib.DistType.FLOAT       
            
        )

        # Add all feature vectors to the index
        self.index.addDataPointBatch(data_matrix)

        # Create/build the HNSW graph
        # M, efConstruction and post are HNSW build parameters
        self.index.createIndex(
            {"M": 16, "efConstruction": 200, "post": 2},
            print_progress=True
        )

        # Mark the index as ready
        self.index_built = True

        # Save the index to disk
        self.index.saveIndex(self.index_file, save_data=True)

        # Save image-path metadata to disk
        with open(self.metadata_file, "wb") as f:
            pickle.dump({"image_paths": self.image_paths}, f)

        print("Index built and saved.")

    def load_index(self):
        # Check whether both index file and metadata file exist
        if not (os.path.exists(self.index_file) and os.path.exists(self.metadata_file)):
            return False

        # Re-create the NMSLIB index object before loading the saved index
        self.index = nmslib.init(
            method="hnsw",
            space="l2",
            data_type=nmslib.DataType.FLOAT
        )

        # Load saved index data from disk
        self.index.loadIndex(self.index_file, load_data=True)

        # Load saved image paths from metadata file
        with open(self.metadata_file, "rb") as f:
            meta = pickle.load(f)
            self.image_paths = meta["image_paths"]

        # Mark the index as ready to use
        self.index_built = True
        print("Index loaded from disk.")
        return True

    def select_query_image(self, query_idx=0):
        # Check that dataset image paths are available
        if not self.image_paths:
            raise ValueError("No images available.")

        # Check that the selected query index is within bounds
        if query_idx < 0 or query_idx >= len(self.image_paths):
            raise IndexError("query_idx out of range.")

        # Return the path of the selected query image
        return self.image_paths[query_idx]

    def search(self, query_path, top_k=10, exclude_query=True):
        # Ensure the HNSW index has been built or loaded
        if not self.index_built:
            raise ValueError("Index not built. Call build_index() or load_index() first.")

        # Extract features from the query image
        query_feat = self.extract_features(query_path)

        # If query image could not be processed, return empty result list
        if query_feat is None:
            return []

        # Ask for one extra result because the query image itself may appear
        k_search = min(top_k + 1, len(self.image_paths))

        # Perform nearest-neighbor search in the HNSW index
        indices, distances = self.index.knnQuery(query_feat, k=k_search)

        # Prepare results list
        results = []

        # Absolute path of the query image, used to exclude the same image if needed
        query_abs = os.path.abspath(query_path)

        # Convert returned neighbor indices into readable search results
        for idx, dist in zip(indices, distances):
            db_path = self.image_paths[idx]

            # Skip the query image itself if exclude_query=True
            if exclude_query and os.path.abspath(db_path) == query_abs:
                continue

            # Store rank placeholder, image path, and distance
            results.append({
                "rank": 0,
                "image_path": db_path,
                "distance": float(dist)
            })

            # Stop once top_k valid results are collected
            if len(results) == top_k:
                break

        # Assign final ranks starting from 1
        for i, r in enumerate(results):
            r["rank"] = i + 1

        return results

    def search_by_dataset_index(self, query_idx=0, top_k=10):
        # Select one image from the dataset as query
        query_path = self.select_query_image(query_idx)

        # Run similarity search
        results = self.search(query_path, top_k=top_k, exclude_query=True)

        # Return query path together with its results
        return query_path, results

    def print_results(self, query_path, results):
        # Print query image path and ranked retrieval results in the console
        print("\n" + "=" * 70)
        print("QUERY IMAGE:")
        print(query_path)

        print("\nMOST SIMILAR IMAGES:")
        for r in results:
            print(f"Rank {r['rank']:2d} | Dist = {r['distance']:.6f} | {r['image_path']}")
        print("=" * 70)

    def save_results_to_folder(self, query_path, results, output_folder="retrieval_results"):
        # Create output folder if it does not exist
        os.makedirs(output_folder, exist_ok=True)

        # Copy the query image into the output folder
        query_name = os.path.basename(query_path)
        shutil.copy2(query_path, os.path.join(output_folder, f"query_{query_name}"))

        # Copy each retrieved image into the output folder with rank and distance in filename
        for r in results:
            src = r["image_path"]
            base = os.path.basename(src)
            dst = os.path.join(
                output_folder,
                f"rank_{r['rank']:02d}_dist_{r['distance']:.4f}_{base}"
            )
            shutil.copy2(src, dst)

        # Save a text file summarizing the retrieval results
        txt_path = os.path.join(output_folder, "results.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"QUERY IMAGE:\n{query_path}\n\n")
            f.write("MOST SIMILAR IMAGES:\n")
            for r in results:
                f.write(f"Rank {r['rank']:2d} | Dist = {r['distance']:.6f} | {r['image_path']}\n")

        print(f"Results saved in folder: {output_folder}")

    def visualize_results(self, query_path, results):
        # Number of retrieved images
        n = len(results)

        # Create a figure with one subplot for query + one for each result
        fig, axes = plt.subplots(1, n + 1, figsize=(4 * (n + 1), 4))

        # Display query image
        query_img = self.read_rgb(query_path)
        axes[0].imshow(query_img)
        axes[0].set_title("QUERY")
        axes[0].axis("off")

        # Display retrieved images with rank and distance
        for i, r in enumerate(results):
            img = self.read_rgb(r["image_path"])
            axes[i + 1].imshow(img)
            axes[i + 1].set_title(f"Rank {r['rank']}\nDist={r['distance']:.4f}")
            axes[i + 1].axis("off")

        # Adjust layout and show figure
        plt.tight_layout()
        plt.show()


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    # Folder containing the dataset images
    dataset_folder = "./images"

    # Create the CBIR system
    cbir = SkinCBIRSystem()

    # Load a subset of images from the dataset
    cbir.load_image_subset(dataset_folder, max_images=10000, shuffle=False)

    # Extract and store feature vectors for all loaded images
    cbir.build_feature_database()

    # Build the HNSW search index
    cbir.build_index()

    # Select one dataset image as query and search for the top 10 most similar images
    query_path, results = cbir.search_by_dataset_index(query_idx=0, top_k=10)

    # Print ranked results in the console
    cbir.print_results(query_path, results)

    # Save query image and retrieved images into a results folder
    cbir.save_results_to_folder(query_path, results, output_folder="retrieval_results")

    # Show query image and retrieved images visually
    cbir.visualize_results(query_path, results)