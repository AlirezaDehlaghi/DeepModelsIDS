import itertools
import json
import os
import numpy as np
import torch
from torch.utils.data import TensorDataset
from Models import train_autoencoder, compute_reconstruction_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve


# Convert dictionary config to a sorted JSON string key
def config_to_key(config):
    return json.dumps(config, sort_keys=True)


# Make objects JSON serializable (e.g. numpy/tensor types to native Python)
def convert_to_serializable(obj):
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_serializable(v) for v in obj)
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, torch.Tensor):
        return obj.tolist()
    else:
        return obj


# Load cached grid search results if available
def load_cache(result_path):
    if result_path and os.path.exists(result_path):
        with open(result_path, "r") as f:
            return json.load(f)
    return {}


# Write updated results to cache
def update_cache(result_path, seen_configs):
    if result_path:
        with open(result_path, "w") as f:
            json.dump(seen_configs, f, indent=2)


# Prepares dataset, including LSTM-specific unfolding if needed
def prepare_dataset(data_tensor, window_size=None, cond_tensor=None):
    if window_size:
        sequences = data_tensor.unfold(0, window_size, 1).permute(0, 2, 1)
        return TensorDataset(sequences)
    return TensorDataset(data_tensor) if cond_tensor is None else TensorDataset(data_tensor, cond_tensor)


# Compute ROC-based threshold and evaluation metrics on the test set
def compute_test_metrics(model, model_class, test_data_tensor, test_labels, test_cond_tensor, window_size, device):
    test_data_tensor_used = test_data_tensor
    if model_class.__name__ == "LSTMAutoEncoder" and window_size:
        test_data_tensor_used = test_data_tensor.unfold(0, window_size, 1).permute(0, 2, 1)
        test_labels = test_labels[window_size-1:]
    test_data_tensor_used = test_data_tensor_used.to(device)
    test_labels_np = np.array(test_labels)

    cond_input = test_cond_tensor.to(device) if model_class.__name__ == "ConditionalVariationalAutoEncoder" else None
    test_errors = compute_reconstruction_error(model, test_data_tensor_used, cond_tensor=cond_input, device=device)

    if not np.all(np.isfinite(test_errors)):
        raise ValueError("❗ test_errors contain NaN or inf values. ")

    fpr, tpr, thresholds = roc_curve(test_labels_np, test_errors)
    best_thresh = float(thresholds[np.argmax(tpr - fpr)])
    preds = (test_errors > best_thresh).astype(int)

    return {
        "threshold": best_thresh,
        "accuracy": float(accuracy_score(test_labels_np, preds)),
        "precision": float(precision_score(test_labels_np, preds)),
        "recall": float(recall_score(test_labels_np, preds)),
        "f1": float(f1_score(test_labels_np, preds))
    }


# Main function for hyperparameter grid search with result caching
def grid_search_autoencoder(
    data_tensor,
    param_grid,
    input_dim,
    model_class,
    device="cpu",
    verbose=True,
    cond_tensor=None,
    test_data_tensor=None,
    test_labels=None,
    test_cond_tensor=None,
    n_samples=1,
    result_path=None
):
    if model_class.__name__ == "ConditionalVariationalAutoEncoder" and cond_tensor is None:
        raise ValueError("Conditional model requires cond_tensor")

    keys = list(param_grid.keys())
    best_score, best_accuracy = float("inf"), -1
    best_config, best_config_by_accuracy = None, None
    all_results = []

    seen_configs = load_cache(result_path)

    for i, values in enumerate(itertools.product(*[param_grid[k] for k in keys]), 1):
        config = dict(zip(keys, values))
        config_key = config_to_key(config)

        # Skip evaluation if already cached
        if config_key in seen_configs:
            if verbose:
                print(f"\n⚠️ Skipping already evaluated config {i}: {config}")
            result = seen_configs[config_key]
            all_results.append((config, result['val_mse'], result.get('metrics', {})))
            if result['val_mse'] < best_score:
                best_score, best_config = result['val_mse'], config
            if 'metrics' in result and result['metrics'].get("accuracy", -1) > best_accuracy:
                best_accuracy, best_config_by_accuracy = result['metrics']['accuracy'], config
            continue

        if verbose:
            print(f"\n🔍 Trying config {i}: {config}")

        # Prepare training/validation split
        window_size = config.get("window_size") if model_class.__name__ == "LSTMAutoEncoder" else None
        dataset = prepare_dataset(data_tensor, window_size, cond_tensor)
        train_size = int(len(dataset) * 0.8)
        train_data = torch.utils.data.Subset(dataset, list(range(train_size)))
        val_data = torch.utils.data.Subset(dataset, list(range(train_size, len(dataset))))

        # Create model instance with appropriate parameters depending on model type
        if model_class.__name__ == "ConditionalVariationalAutoEncoder":
            model = model_class(
                input_dim=input_dim,
                cond_dim=cond_tensor.shape[1],
                hidden_layers=config["hidden_layers"],
                latent_dim=config["latent_dim"],
                dropout=config.get("dropout", 0.0),
                activation=config.get("activation", "relu")
            )
        elif model_class.__name__ == "LSTMAutoEncoder":
            model = model_class(
                input_dim=input_dim,
                hidden_dim=config["hidden_dim"],
                latent_dim=config["latent_dim"],
                num_layers=config.get("num_layers", 1)
            )
        else:
            model = model_class(
                input_dim=input_dim,
                hidden_layers=config["hidden_layers"],
                latent_dim=config["latent_dim"],
                dropout=config.get("dropout", 0.0),
                activation=config.get("activation", "relu")
            )
        # Train model
        model = train_autoencoder(
            model=model,
            train_data=train_data,
            batch_size=config["batch_size"],
            lr=config["lr"],
            optimizer_name=config["optimizer"],
            num_epochs=config["epochs"],
            device=device,
            verbose=verbose,
            n_samples=n_samples
        )

        # Compute validation error
        val_inputs = torch.stack([x[0] for x in val_data])
        val_conds = torch.stack([x[1] for x in val_data]) if model_class.__name__ == "ConditionalVariationalAutoEncoder" else None
        val_errors = compute_reconstruction_error(model, val_inputs, cond_tensor=val_conds, device=device)
        avg_error = float(np.mean(val_errors))

        # Evaluate on test set
        best_metrics = {}
        if test_data_tensor is not None and test_labels is not None:
            best_metrics = compute_test_metrics(model, model_class, test_data_tensor, test_labels, test_cond_tensor, window_size, device)
            if best_metrics["accuracy"] > best_accuracy:
                best_accuracy, best_config_by_accuracy = best_metrics["accuracy"], config

            if verbose:
                print(f"📈 Threshold: {best_metrics['threshold']:.4f} | Accuracy: {best_metrics['accuracy']:.4f} | Precision: {best_metrics['precision']:.4f} | Recall: {best_metrics['recall']:.4f} | F1: {best_metrics['f1']:.4f}")

        # Record result
        all_results.append((config, avg_error, best_metrics))
        if avg_error < best_score:
            best_score, best_config = avg_error, config

        if verbose:
            print(f"✅ Validation MSE: {avg_error:.6f}")

        # Update result cache
        seen_configs[config_key] = {
            "val_mse": convert_to_serializable(avg_error),
            "metrics": {k: convert_to_serializable(v) for k, v in best_metrics.items()}
        }
        update_cache(result_path, seen_configs)

    # Print summary
    print("\n🏆 Best Config:", best_config)
    print(f"Lowest Validation MSE: {best_score:.6f}")
    if best_config_by_accuracy is not None:
        print("\n🎯 Best Config by Test Accuracy:", best_config_by_accuracy)
        print(f"Highest Test Accuracy: {best_accuracy:.4f}")

    return best_config, best_config_by_accuracy, all_results
