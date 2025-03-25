import numpy as np
import urllib.request
import os
from scipy.ndimage import zoom
from PIL import Image

# --------------------------
# Pattern Generation Helpers
# --------------------------

def generate_random_patterns(shape, n_patterns):
    """
    Generate a list of random patterns.
    
    Args:
        dimension (int): Dimension of each pattern.
        n_patterns (int): Number of patterns.
    
    Returns:
        list of np.array: List of random patterns.
    """
    return [np.random.choice([1, -1], size=shape) for _ in range(n_patterns)]

def corrupt_pattern_random(pattern, q):
    """
    Corrupt a pattern randomly: flip each pixel with probability (1-q).
    
    Args:
        pattern (np.array): Original pattern.
        q (float): Probability to keep each pixel unchanged.
    
    Returns:
        np.array: Corrupted pattern.
    """
    corrupted = pattern.flatten()
    flip_mask = np.random.rand(len(corrupted)) >= q
    corrupted[flip_mask] *= -1
    return corrupted.reshape(pattern.shape)

def corrupt_pattern_by_row(pattern, q):
    """
    Corrupt a pattern row by row: for each row, with probability (1-q)
    flip the entire row.
    
    Args:
        pattern (np.array): Original 2D pattern.
        q (float): Probability to keep the row unchanged.
    
    Returns:
        np.array: Corrupted pattern.
    """
    corrupted = pattern.copy()
    for i in range(corrupted.shape[0]):
        if np.random.rand() >= q:
            corrupted[i, :] *= -1
    return corrupted

def corrupt_pattern_by_column(pattern, q):
    """
    Corrupt a pattern column by column: for each column, with probability (1-q)
    flip the entire column.
    
    Args:
        pattern (np.array): Original 2D pattern.
        q (float): Probability to keep the column unchanged.
    
    Returns:
        np.array: Corrupted pattern.
    """
    corrupted = pattern.copy()
    for j in range(corrupted.shape[1]):
        if np.random.rand() >= q:
            corrupted[:, j] *= -1
    return corrupted

def corrupt_pattern_gradual(pattern, q, direction='vertical', width_fraction=0.1):
    """
    Corrupt a 2D pattern gradually along the specified direction.
    
    For 'vertical': the top part of the image is preserved and the bottom is fully corrupted,
    with a linear transition band of height = width_fraction*H centered at row = q*H.
    
    Args:
        pattern (np.array): Original 2D pattern.
        q (float): Fraction of rows (if vertical) or columns (if horizontal) that remain uncorrupted 
                   (in the center of the transition the probability decreases linearly).
        direction (str): 'vertical' or 'horizontal'
        width_fraction (float): Fraction of total dimension that defines the width of the transition band.
    
    Returns:
        np.array: Corrupted pattern.
    """
    corrupted = pattern.copy()
    if direction == 'vertical':
        H, W = pattern.shape
        # Calculate transition band
        trans_width = int(H * width_fraction)
        # The center of the transition is at the corresponding row q*H
        center = int(q * H)
        start = max(0, center - trans_width // 2)
        end = min(H, center + trans_width // 2)
        for i in range(H):
            if i < start:
                p_keep = 1.0
            elif i > end:
                p_keep = 0.0
            else:
                # Linear transition from 1 to 0
                p_keep = 1.0 - (i - start) / (end - start)
            # Applica la corruzione a livello di riga: ogni pixel della riga viene flipato con probabilità (1 - p_keep)
            flip_mask = np.random.rand(W) >= p_keep
            # Qui, per "corruzione graduale", forziamo il pixel a -1 (oscuramento)
            corrupted[i, flip_mask] = -1
        return corrupted
    elif direction == 'horizontal':
        H, W = pattern.shape
        trans_width = int(W * width_fraction)
        center = int(q * W)
        start = max(0, center - trans_width // 2)
        end = min(W, center + trans_width // 2)
        for j in range(W):
            if j < start:
                p_keep = 1.0
            elif j > end:
                p_keep = 0.0
            else:
                p_keep = 1.0 - (j - start) / (end - start)
            flip_mask = np.random.rand(H) >= p_keep
            corrupted[flip_mask, j] = -1
        return corrupted
    else:
        raise ValueError("Direction must be 'vertical' or 'horizontal'.")

def corrupt_patterns(patterns, q, method="random", **kwargs):
    """
    Apply corruption to a list of patterns using different methods.
    
    Args:
        patterns (list of np.array): List of original patterns.
        q (float): Parameter for corruption (interpretation depends on method).
        method (str): One of 'random', 'row', 'column', 'gradual'.
        **kwargs: Additional parameters (e.g. direction, width_fraction for 'gradual').
    
    Returns:
        np.array: Array of corrupted patterns.
    """
    corrupted_list = []
    for pattern in patterns:
        if method == "random":
            # Usa la versione esistente: flip a livello di pixel randomicamente
            corrupted_list.append(corrupt_pattern_random(pattern, q))
        elif method == "row":
            # Per ogni riga, con probabilità (1-q) flipa l'intera riga
            corrupted = pattern.copy()
            for i in range(corrupted.shape[0]):
                if np.random.rand() >= q:
                    corrupted[i, :] *= -1
            corrupted_list.append(corrupted)
        elif method == "column":
            # Per ogni colonna, con probabilità (1-q) flipa l'intera colonna
            corrupted = pattern.copy()
            for j in range(corrupted.shape[1]):
                if np.random.rand() >= q:
                    corrupted[:, j] *= -1
            corrupted_list.append(corrupted)
        elif method == "gradual":
            # Usa la nuova funzione: kwargs possono contenere "direction" e "width_fraction"
            corrupted_list.append(corrupt_pattern_gradual(pattern, q, **kwargs))
        else:
            raise ValueError("Unknown corruption method. Use 'random', 'row', 'column', or 'gradual'.")
    return np.array(corrupted_list)



def load_mnist_npz():
    url = "https://s3.amazonaws.com/img-datasets/mnist.npz"
    filename = "mnist.npz"
    if not os.path.exists(filename):
        print("Downloading MNIST...")
        urllib.request.urlretrieve(url, filename)
    data = np.load(filename)
    return data["x_train"], data["y_train"]
    
def get_mnist_patterns(scale=1):
    """
    Takes (0-9) numbers binary them and superscale 
    the resolution with nearest-neighbor interpolation
    """
    x_train, y_train = load_mnist_npz()
    patterns = {}
    
    for img, label in zip(x_train, y_train):
        if label not in patterns:
            binary_img = np.where(img > 0.5, 1, -1)  # Prima binarizzazione
            upscaled_img = zoom(binary_img, scale, order=0)  # Nearest-neighbor
            patterns[label] = upscaled_img
        
        if len(patterns) == 10:
            break

    return [patterns[i] for i in range(10)]

def image_to_pattern(image_path, threshold=128, size=(150, 150)):
    """
    Convert a single image to a 2D binary pattern.

    Args:
        image_path (str): Path to the image file.
        threshold (int, optional): Threshold for binarizing the image.
        size (tuple, optional): Desired image size (width, height).

    Returns:
        np.ndarray: Binary pattern with values 1 (for pixels > threshold) and -1 (for pixels <= threshold).
    """
    # Load the image
    img = Image.open(image_path)
    # Resize the image to the desired dimensions
    img = img.resize(size)
    # Convert the image to grayscale
    gray_img = img.convert('L')
    # Convert the grayscale image to a numpy array
    img_array = np.array(gray_img)
    # Binarize the image: pixels > threshold become 1, otherwise -1
    binary_pattern = np.where(img_array > threshold, 1, -1)
    return binary_pattern

def get_images_patterns(directory="images_patterns", max_images=None, threshold=128, size=(150, 150)):
    """
    Load image files from the specified directory, convert each to a binary pattern using image_to_pattern.

    Args:
        directory (str): Directory containing image files.
        max_images (int, optional): Maximum number of images to load. If None, load all images.
        threshold (int, optional): Threshold for binarizing images.
        size (tuple, optional): Size (width, height) to which images are resized.

    Returns:
        list: List of binary patterns (numpy arrays) for the loaded images.
    """
    patterns = []
    # List all files in the directory and sort them (so the order is deterministic)
    files = sorted(os.listdir(directory))
    
    # If max_images is specified, take only the first max_images files
    if max_images is not None:
        files = files[:max_images]
    
    # Valid image file extensions
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
    
    # Process each file in the directory
    for file in files:
        if file.lower().endswith(valid_extensions):
            image_path = os.path.join(directory, file)
            pattern = image_to_pattern(image_path, threshold=threshold, size=size)
            patterns.append(pattern)
    
    return patterns
