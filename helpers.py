import matplotlib.pyplot as plt

def plot_mnist_results(original, corrupted, recovered, digit, figsize=(6,2)):
    """
    Plot the original, corrupted, and recovered MNIST images for a given digit.
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title(f"Original {digit}")
    axes[0].axis("off")
    
    axes[1].imshow(corrupted, cmap='gray')
    axes[1].set_title(f"Corrupted {digit}")
    axes[1].axis("off")
    
    axes[2].imshow(recovered, cmap='gray')
    axes[2].set_title(f"Recovered {digit}")
    axes[2].axis("off")
    
    plt.tight_layout()
    plt.show()

def plot_all_results(patterns, corrupted_patterns, recovered_patterns, filepath, model=None, figsize = (9,3) ):
    """
    Plot original, corrupted, and recovered images for all patterns.

    Args:
        patterns (iterable): Collection of original images.
        corrupted_patterns (iterable): Collection of corrupted images.
        recovered_patterns (iterable): Collection of recovered images.
        model (object, optional): Hopfield model with an `overlap` method. 
                                  If provided, computes and displays the overlap.
    """
    for i, (orig, corr, recov) in enumerate(zip(patterns, corrupted_patterns, recovered_patterns)):
        fig, axes = plt.subplots(1, 3, figsize= figsize)
        
        # Plot original pattern
        axes[0].imshow(orig, cmap='gray')
        axes[0].set_title(f"Original {i}")
        axes[0].axis("off")
        
        # Plot corrupted pattern
        axes[1].imshow(corr, cmap='gray')
        axes[1].set_title(f"Corrupted {i}")
        axes[1].axis("off")
        
        # Plot recovered pattern and display overlap if model is provided
        if model is not None:
            overlap = model.overlap(orig, recov)
            title_rec = f"Recovered {i}\nOverlap: {overlap:.3f}"
        else:
            title_rec = f"Recovered {i}"
        axes[2].imshow(recov, cmap='gray')
        axes[2].set_title(title_rec)
        axes[2].axis("off")
        
        plt.tight_layout()
        plt.savefig(filepath)
        plt.show()


import itertools
import concurrent.futures
from tqdm import tqdm
import pandas as pd
import numpy as np
from HopfieldModel import HopfieldModelnD
from patterns import generate_random_patterns, corrupt_patterns, get_mnist_patterns
from functools import partial


def grid_search_random_task(params, patterns=None, corruption_function=None):
    """
    Execute a grid search task for a combination of parameters on given patterns.
    
    Args:
        params (tuple): (q, n_patterns, dim, T, alpha, schedule, R, update_method, learning_rule)
        patterns (optional): Pre-generated patterns. If None, generate them using generate_random_patterns.
        corruption_function (optional): Custom function to corrupt patterns.
            Should accept (patterns, q) and return corrupted patterns.
    
    Returns:
        dict: Dictionary with all input parameters and computed metrics (avg_overlap, std_overlap, interference, energy_diff),
              as well as the model and parameters used.
              (Note: recovered patterns are not saved.)
    """
    q, n_patterns, dim, T, alpha, schedule, R, update_method, learning_rule = params

    # If patterns are not provided, generate them.
    # Assumes generate_random_patterns accepts a shape (dim) and number of patterns.
    if patterns is None:
        patterns = generate_random_patterns(dim, n_patterns)
    
    # Use the provided corruption_function if available, otherwise use default corrupt_patterns.
    if corruption_function is None:
        corrupted = corrupt_patterns(patterns, q)
    else:
        corrupted = corruption_function(patterns, q)
    
    # Create the Hopfield model with the specified parameters.
    model = HopfieldModelnD(patterns, update_method=update_method, learning_rule=learning_rule, R=R, verbose=False)
    
    # Recover the patterns.
    recovered = model.correct_patterns(corrupted, max_iter=100, convergence_check=1, temperature=T, alpha=alpha, schedule=schedule)
    
    # Compute metrics.
    overlaps = [model.overlap(patterns[i], recovered[i]) for i in range(n_patterns)]
    avg_overlap = np.mean(overlaps)
    std_overlap = np.std(overlaps)
    #std_overlap = np.sqrt((avg_overlap * (1 - avg_overlap)) / dim)
    interference = model.memory_interference()
    E_diff = model.energy(recovered[0]) - model.energy(patterns[0])
    
    # Return a dictionary with all input parameters and computed metrics.
    return {
        'q': q,
        'n_patterns': n_patterns,
        'dim': dim,
        'temperature': T,
        'alpha': alpha,
        'schedule': schedule,
        'R': R,
        'update_method': update_method,
        'learning_rule': learning_rule,
        'avg_overlap': avg_overlap,
        'std_overlap': std_overlap,
        'interference': interference,
        'energy_diff': E_diff,
        'model': model
    }

def grid_search_parallel(param_grid, save_path=None, use_progress=True, n_jobs=None, best_metric_key=None, **task_kwargs):
    """
    Perform a parallel grid search using grid_search_random_task.
    
    Args:
        param_grid (dict): Dictionary mapping parameter names to possible values.
            IMPORTANT: For constant parameters (e.g. 'dim', 'R'), pass them as scalars or wrap in a list.
            For iterated parameters (e.g. 'learning_rule', 'update_method', 'schedule'), pass as lists.
        save_path (str, optional): Path to save the results CSV.
        use_progress (bool, optional): If True, show a progress bar.
        n_jobs (int, optional): Number of processes to use.
        best_metric_key (str, optional): Key used to select the best model (e.g. 'avg_overlap').
        **task_kwargs: Additional keyword arguments passed to grid_search_random_task 
                       (e.g. patterns, corruption_function).
    
    Returns:
        tuple: (DataFrame, best_params)
            - DataFrame: All grid search results.
            - best_params: Dictionary with the parameters of the best model (based on best_metric_key).
    """
    # Convert each parameter:
    for key, value in param_grid.items():
        if isinstance(value, np.ndarray):
            param_grid[key] = list(value)
        elif not isinstance(value, list):
            param_grid[key] = [value]
    
    # Use natural order of keys.
    param_order = list(param_grid.keys())
    
    # Generate all parameter combinations as tuples.
    combos = list(itertools.product(*(param_grid[key] for key in param_order)))
    
    results = []
    
    # Use functools.partial to create a pickleable worker function.
    worker = partial(grid_search_random_task, **task_kwargs)
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs) as executor:
        tasks = executor.map(worker, combos)
        if use_progress:
            tasks = tqdm(tasks, total=len(combos), desc="Grid Search Progress")
        for res in tasks:
            results.append(res)
    
    # Create a DataFrame from the results and optionally save.
    df = pd.DataFrame(results)
    if save_path:
        df.to_csv(save_path, index=False)
    
    best_params = None
    if best_metric_key and best_metric_key in df.columns:
        best_row = df.loc[df[best_metric_key].idxmax()]
        best_params = best_row.to_dict()
    
    return df, best_params
