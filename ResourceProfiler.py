import os
import gc
import time
import psutil
import threading
import pandas as pd
from os import path

from torch.utils.data import TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve

from Models import *
from Preprocessor import preprocess_data
from GridSearch import convert_to_serializable
from pympler import asizeof

from HelperFucntions import *


def prepare_dataset(data_tensor, window_size=None, cond_tensor=None):
    if window_size:
        sequences = data_tensor.unfold(0, window_size, 1).permute(0, 2, 1)
        return TensorDataset(sequences)
    return TensorDataset(data_tensor) if cond_tensor is None else TensorDataset(data_tensor, cond_tensor)


def read_data(dataset_path, model_class, window_size):
    Report.green("Loading data...[started]")
    data = pd.read_csv(dataset_path)
    x_train, x_val, x_test, y_train, y_val, y_test = preprocess_data(
        data=data, is_binary=True, is_supervised=False, train_portion=0.38, validation_portion=0.16
    )

    X_train, X_val, y_val, X_test, y_test = (
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(x_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.int32),
        torch.tensor(x_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.int32),
    )

    if model_class.__name__ == "ConditionalVariationalAutoEncoder":
        x_train_final, c_train_final = X_train[:, :47], X_train[:, 47:]
        x_val_final, c_val_final = X_val[:, :47], X_val[:, 47:]
        x_test_final, c_test_final = X_test[:, :47], X_test[:, 47:]
    else:
        x_train_final, c_train_final = X_train, None
        x_val_final, c_val_final = X_val, None
        x_test_final, c_test_final = X_test, None

    if model_class.__name__ == "LSTMAutoEncoder":
        y_val = y_val[window_size - 1:]
        y_test = y_test[window_size - 1:]
    Report.green("Loading data...[end]")
    return (
        prepare_dataset(x_train_final, window_size, c_train_final),
        prepare_dataset(x_val_final, window_size, c_val_final),
        prepare_dataset(x_test_final, window_size, c_test_final),
        y_val, y_test
    )


def compute_test_metrics(model, x_val, y_val, x_test, y_test, cond_val, cond_test, device):
    model.eval()
    x_val_used = x_val.to(device)
    x_test_used = x_test.to(device)
    cond_val = cond_val.to(device) if cond_val is not None else None
    cond_test = cond_test.to(device) if cond_test is not None else None

    val_errors = compute_reconstruction_error(model, x_val_used, cond_tensor=cond_val, device=device)
    fpr, tpr, thresholds = roc_curve(y_val.cpu().numpy(), val_errors)
    best_thresh = float(thresholds[np.argmax(tpr - fpr)])

    test_errors = compute_reconstruction_error(model, x_test_used, cond_tensor=cond_test, device=device)
    preds = (test_errors > best_thresh).astype(int)

    return {
        "Accuracy": accuracy_score(y_test, preds),
        "Precision": precision_score(y_test, preds),
        "Recall": recall_score(y_test, preds),
        "F1": f1_score(y_test, preds),
        "AUC": roc_auc_score(y_test.cpu().numpy(), test_errors),
        "Threshold": best_thresh
    }


def monitor_usage(proc, stop_flag, cpu_samples, mem_samples, interval):
    while not stop_flag[0]:
        cpu_samples.append(proc.cpu_percent(interval=None))
        mem_samples.append(proc.memory_info().rss)
        time.sleep(interval)


def profile_train(model_class, config, train_data):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    proc = psutil.Process(os.getpid())

    x_train = torch.stack([x[0] for x in train_data])
    c_train = torch.stack([x[1] for x in train_data]) if model_class.__name__ == "ConditionalVariationalAutoEncoder" else None
    input_dim = x_train.shape[-1]
    cond_dim = c_train.shape[1] if c_train is not None else None

    if model_class.__name__ == "ConditionalVariationalAutoEncoder":
        model = model_class(input_dim, cond_dim, config["hidden_layers"], config["latent_dim"], config.get("dropout", 0.0), config.get("activation", "relu"))
    elif model_class.__name__ == "LSTMAutoEncoder":
        model = model_class(input_dim, config["hidden_dim"], config["latent_dim"], config.get("num_layers", 1))
    else:
        model = model_class(input_dim, config["hidden_layers"], config["latent_dim"], config.get("dropout", 0.0), config.get("activation", "relu"))

    model.to(device)

    cpu_samples, mem_samples, stop_flag = [], [], [False]
    t = threading.Thread(target=monitor_usage, args=(proc, stop_flag, cpu_samples, mem_samples, 1))
    t.start()

    start = time.perf_counter()
    model = train_autoencoder(
        model=model,
        train_data=train_data,
        batch_size=config["batch_size"],
        lr=config["lr"],
        optimizer_name=config["optimizer"],
        num_epochs=config["epochs"],
        device=device,
        verbose=True
    )
    duration = time.perf_counter() - start

    stop_flag[0] = True
    t.join()

    model_path = path.join("TrainedModels", f"{model_class.__name__}_trained.pt")
    torch.save(model, model_path)

    model_size_ram = asizeof.asizeof(model) / 1024
    model_size_disk = os.path.getsize(model_path) / 1024
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    train_ram = (mem_samples[-1] - mem_samples[0]) / (1024 ** 2)
    peak_ram = (max(mem_samples) - mem_samples[0]) / (1024 ** 2)
    cpu_avg = np.mean(cpu_samples)  / psutil.cpu_count(logical=True)

    return {
        "Model": model_class.__name__,
        "TrainTime_sec": round(duration, 2),
        "TrainRAM_MB": round(train_ram, 2),
        "TrainPeakRAM_MB": round(peak_ram, 2),
        "ModelRAM_KB": round(model_size_ram, 2),
        "DiskSize_KB": round(model_size_disk, 2),
        "Params": n_params,
        "CPU_Usage_percent": round(cpu_avg, 2)
    }


def profile_test(model_class, config, val_data, test_data, y_val, y_test):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    proc = psutil.Process(os.getpid())

    model_path = path.join("TrainedModels", f"{model_class.__name__}_trained.pt")
    model = torch.load(model_path, weights_only=False)
    model.to(device)
    model.eval()

    x_val = torch.stack([x[0] for x in val_data])
    c_val = torch.stack([x[1] for x in val_data]) if model_class.__name__ == "ConditionalVariationalAutoEncoder" else None
    x_test = torch.stack([x[0] for x in test_data])
    c_test = torch.stack([x[1] for x in test_data]) if model_class.__name__ == "ConditionalVariationalAutoEncoder" else None

    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(1)

    cpu_samples, mem_samples, stop_flag = [], [], [False]
    t = threading.Thread(target=monitor_usage, args=(proc, stop_flag, cpu_samples, mem_samples, 0.02))
    t.start()

    start = time.perf_counter()
    metrics = compute_test_metrics(model, x_val, y_val, x_test, y_test, c_val, c_test, device)
    duration = (time.perf_counter() - start) * 1000

    stop_flag[0] = True
    t.join()

    inference_ram = (mem_samples[-1] - mem_samples[0]) / (1024 ** 2)
    peak_ram = (max(mem_samples) - mem_samples[0]) / (1024 ** 2)
    cpu_avg = np.mean(cpu_samples) / psutil.cpu_count(logical=True)


    return {
        "Model": model_class.__name__,
        "InferenceTime_ms": round(duration, 2),
        "InferenceRAM_MB": round(inference_ram, 2),
        "InferencePeakRAM_MB": round(peak_ram, 2),
        "CPU_Usage_percent": round(cpu_avg, 2),
        **{k: round(v, 4) for k, v in metrics.items()}
    }


if __name__ == "__main__":
    import argparse
    from Models import AutoEncoder, VariationalAutoEncoder, ConditionalVariationalAutoEncoder, LSTMAutoEncoder, ProbabilisticVariationalEncoder

    model_classes = {
        "AutoEncoder": AutoEncoder,
        "VariationalAutoEncoder": VariationalAutoEncoder,
        "ConditionalVariationalAutoEncoder": ConditionalVariationalAutoEncoder,
        "LSTMAutoEncoder": LSTMAutoEncoder,
        "ProbabilisticVariationalEncoder": ProbabilisticVariationalEncoder
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["train", "test"], help="Phase: train or test")
    parser.add_argument("model_name", choices=model_classes.keys(), help="Model class name")
    args = parser.parse_args()

    Report.green(".")
    Report.green(f" Experiment started for {args.phase}ing of {args.model_name}")
    model_class = model_classes[args.model_name]

    base_config = {
        "hidden_layers": [64, 32],
        "latent_dim": 10,
        "dropout": 0.1,
        "activation": "relu",
        "batch_size": 32,
        "lr": 1e-3,
        "epochs": 50,
        "optimizer": "adam",
    } if model_class.__name__ != "LSTMAutoEncoder" else {
        "hidden_dim": 32,
        "latent_dim": 10,
        "num_layers": 1,
        "batch_size": 32,
        "lr": 1e-3,
        "optimizer": "adam",
        "epochs": 50,
        "window_size": 5,
    }

    BEST_CONFIG = {
        "AutoEncoder": {
            "hidden_layers": [64],
            "latent_dim": 30,
            "dropout": 0.0,
            "activation": "leaky_relu",
            "batch_size": 32,
            "lr": 1e-3,
            "epochs": 50,
            "optimizer": "adam"
        },

        "VariationalAutoEncoder": {
            "hidden_layers": [32],
            "latent_dim": 30,
            "dropout": 0.1,
            "activation": "relu",
            "batch_size": 32,
            "lr": 1e-3,
            "epochs": 50,
            "optimizer": "adam"
        },

        "ConditionalVariationalAutoEncoder": {
            "hidden_layers": [64],
            "latent_dim": 20,
            "dropout": 0.1,
            "activation": "relu",
            "batch_size": 32,
            "lr": 1e-3,
            "epochs": 50,
            "optimizer": "adam"
        },

        "ProbabilisticVariationalEncoder": {
            "hidden_layers": [128, 64],
            "latent_dim": 20,
            "dropout": 0.0,
            "activation": "leaky_relu",
            "batch_size": 32,
            "lr": 1e-3,
            "epochs": 50,
            "optimizer": "adam"
        },



        "LSTMAutoEncoder": {
            "hidden_dim": 128,
            "latent_dim": 30,
            "num_layers": 1,
            "batch_size": 32,
            "lr": 1e-3,
            "epochs": 50,
            "optimizer": "adam",
            "window_size": 5
        }
    }

    BEST_CONFIG_ACC = {
        "AutoEncoder": {
            "hidden_layers": [64],
            "latent_dim": 30,
            "dropout": 0.0,
            "activation": "leaky_relu",
            "batch_size": 32,
            "lr": 1e-3,
            "epochs": 50,
            "optimizer": "adam"
        },

        "VariationalAutoEncoder": {
            "hidden_layers": [128, 64],
            "latent_dim": 30,
            "dropout": 0.0,
            "activation": "relu",
            "batch_size": 32,
            "lr": 1e-3,
            "epochs": 50,
            "optimizer": "adam"
        },

        "ConditionalVariationalAutoEncoder": {
            "hidden_layers": [64, 32],
            "latent_dim": 30,
            "dropout": 0.1,
            "activation": "leaky_relu",
            "batch_size": 32,
            "lr": 1e-3,
            "epochs": 50,
            "optimizer": "sgd"
        },



        "ProbabilisticVariationalEncoder": {
            "hidden_layers": [128, 64],
            "latent_dim": 10,
            "dropout": 0.0,
            "activation": "leaky_relu",
            "batch_size": 32,
            "lr": 1e-3,
            "epochs": 50,
            "optimizer": "adam"
        },

        "LSTMAutoEncoder": {
            "hidden_dim": 64,
            "latent_dim": 10,
            "num_layers": 2,
            "batch_size": 32,
            "lr": 1e-3,
            "epochs": 50,
            "optimizer": "adam",
            "window_size": 5
        }
    }

    base_config = BEST_CONFIG[args.model_name]
    Report.cyan(f" Config: {base_config}")

    window_size = base_config.get("window_size") if model_class.__name__ == "LSTMAutoEncoder" else None
    dataset_path = path.join('Dataset', 'Dataset.csv')
    train_final, val_final, test_final, y_val, y_test = read_data(dataset_path, model_class, window_size)

    if args.phase == "train":
        result = profile_train(model_class, base_config, train_final)
        output_csv_path = "Train_runs.csv"
    else:
        result = profile_test(model_class, base_config, val_final, test_final, y_val, y_test)
        output_csv_path = "Test_runs.csv"

    if os.path.exists(output_csv_path):
        existing_df = pd.read_csv(output_csv_path)
        df = pd.concat([existing_df, pd.DataFrame([result])], ignore_index=True)
    else:
        df = pd.DataFrame([result])

    df.to_csv(output_csv_path, index=False)
    Report.green(f"Results saved to {output_csv_path}")


# # ResourceProfiler.py
# import os
# import gc
# import json
# import sys
# import time
# import psutil
# import torch
# import tempfile
# import numpy as np
# import pandas as pd
# from os import path
#
# from torch.cuda import memory_usage
# from torch.utils.data import DataLoader, TensorDataset
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score
#
# from Models import *
# from HelperFucntions import Report
# from Preprocessor import preprocess_data
# from GridSearch import convert_to_serializable
# from pympler import asizeof
#
#
# def prepare_dataset(data_tensor, window_size=None, cond_tensor=None):
#     if window_size:
#         sequences = data_tensor.unfold(0, window_size, 1).permute(0, 2, 1)
#         return TensorDataset(sequences)
#     return TensorDataset(data_tensor) if cond_tensor is None else TensorDataset(data_tensor, cond_tensor)
#
#
# def read_data(dataset_path, model_class, window_size):
#     Report.green(f'Reading dataset for is_binary=True and is_supervised={False} started ...!')
#     data = pd.read_csv(dataset_path)
#     x_train, x_val, x_test, y_train, y_val, y_test = preprocess_data(
#         data=data, is_binary=True, is_supervised=False, train_portion=0.38, validation_portion=0.16
#     )
#
#     X_train, X_val, y_val, X_test, y_test = (
#         torch.tensor(x_train, dtype=torch.float32),
#         torch.tensor(x_val, dtype=torch.float32),
#         torch.tensor(y_val, dtype=torch.int32),
#         torch.tensor(x_test, dtype=torch.float32),
#         torch.tensor(y_test, dtype=torch.int32),
#     )
#
#     if model_class.__name__ == "ConditionalVariationalAutoEncoder":
#         x_train_final, c_train_final = X_train[:, :47], X_train[:, 47:]
#         x_val_final, c_val_final = X_val[:, :47], X_val[:, 47:]
#         x_test_final, c_test_final = X_test[:, :47], X_test[:, 47:]
#     else:
#         x_train_final, c_train_final = X_train, None
#         x_val_final, c_val_final = X_val, None
#         x_test_final, c_test_final = X_test, None
#
#     if model_class.__name__ == "LSTMAutoEncoder":
#         y_val = y_val[window_size-1:]
#         y_test = y_test[window_size-1:]
#
#     return (
#         prepare_dataset(x_train_final, window_size, c_train_final),
#         prepare_dataset(x_val_final, window_size, c_val_final),
#         prepare_dataset(x_test_final, window_size, c_test_final),
#         y_val, y_test
#     )
#
#
# def compute_test_metrics(model, x_val, y_val, x_test, y_test, cond_val, cond_test, device):
#     model.eval()
#     x_val_used = x_val.to(device)
#     x_test_used = x_test.to(device)
#     cond_val = cond_val.to(device) if cond_val is not None else None
#     cond_test = cond_test.to(device) if cond_test is not None else None
#
#     val_errors = compute_reconstruction_error(model, x_val_used, cond_tensor=cond_val, device=device)
#     fpr, tpr, thresholds = roc_curve(y_val.cpu().numpy(), val_errors)
#     best_thresh = float(thresholds[np.argmax(tpr - fpr)])
#
#     test_errors = compute_reconstruction_error(model, x_test_used, cond_tensor=cond_test, device=device)
#     preds = (test_errors > best_thresh).astype(int)
#     auc_score = roc_auc_score(y_test.cpu().numpy(), test_errors)
#
#     return {
#         "Test_Threshold": best_thresh,
#         "Test_AUC": auc_score,
#         "Test_Accuracy": accuracy_score(y_test, preds),
#         "Test_Precision": precision_score(y_test, preds),
#         "Test_Recall": recall_score(y_test, preds),
#         "Test_F1": f1_score(y_test, preds)
#     }
#
#
# def run_inference(model, xb, cb):
#     with torch.no_grad():
#         _ = model(xb, cb) if cb is not None else model(xb)
#
#
# def profile_model(model_class, config, train_data, val_data, test_data, y_val, y_test):
#
#     model_path = path.join("TrainedModels", f"{model_class.__name__}_trained.pt")
#
#     # initial steps to prepare data
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     x_train = torch.stack([x[0] for x in train_data])
#     c_train = torch.stack([x[1] for x in train_data]) if model_class.__name__ == "ConditionalVariationalAutoEncoder" else None
#     x_val = torch.stack([x[0] for x in val_data])
#     c_val = torch.stack([x[1] for x in val_data]) if model_class.__name__ == "ConditionalVariationalAutoEncoder" else None
#     x_test = torch.stack([x[0] for x in test_data])
#     c_test = torch.stack([x[1] for x in test_data]) if model_class.__name__ == "ConditionalVariationalAutoEncoder" else None
#
#     input_dim = x_train.shape[-1]
#     cond_dim = c_train.shape[1] if c_train is not None else None
#
#     load_available = model_path and os.path.exists(model_path)
#
#     # saving the state before training
#     gc.collect()
#     torch.cuda.empty_cache()
#     time.sleep(1)
#     proc = psutil.Process(os.getpid())
#
#     # Train the model
#
#     mem_before_train = proc.memory_info().rss
#     t0 = time.perf_counter()
#     if load_available:
#         Report.cyan("Loading model from file (skip training)...")
#         model = torch.load(model_path, weights_only=False)
#     else:
#         Report.cyan("Training model...")
#         # Model initialization
#         if model_class.__name__ == "ConditionalVariationalAutoEncoder":
#             model = model_class(
#                 input_dim=input_dim,
#                 cond_dim=cond_dim,
#                 hidden_layers=config["hidden_layers"],
#                 latent_dim=config["latent_dim"],
#                 dropout=config.get("dropout", 0.0),
#                 activation=config.get("activation", "relu")
#             )
#         elif model_class.__name__ == "LSTMAutoEncoder":
#             model = model_class(
#                 input_dim=input_dim,
#                 hidden_dim=config["hidden_dim"],
#                 latent_dim=config["latent_dim"],
#                 num_layers=config.get("num_layers", 1)
#             )
#         else:
#             model = model_class(
#                 input_dim=input_dim,
#                 hidden_layers=config["hidden_layers"],
#                 latent_dim=config["latent_dim"],
#                 dropout=config.get("dropout", 0.0),
#                 activation=config.get("activation", "relu")
#             )
#
#         model.to(device)
#         model = train_autoencoder(model, train_data, config["batch_size"], config["lr"],
#                                   config["optimizer"], config["epochs"], device, verbose=True)
#         if model_path:
#             torch.save(model, model_path)
#     model.to(device)
#     t_train = time.perf_counter() - t0
#     mem_after_train = proc.memory_info().rss
#     train_ram = (mem_after_train - mem_before_train) / (1024 * 1)
#     Report.blue(f"Train RAM: {train_ram}")
#
#     # Validation error
#     val_x = x_val.to(device)
#     val_c = c_val.to(device) if c_val is not None else None
#     val_err = compute_reconstruction_error(model, val_x, cond_tensor=val_c, device=device).mean()
#
#     # Inference RAM profiling
#     tmp_model_path = tempfile.NamedTemporaryFile(delete=False)
#     tmp_model_path.close()
#     torch.save(model, tmp_model_path.name)
#
#     del model, train_data, val_data, x_train, c_train
#     gc.collect()
#     torch.cuda.empty_cache()
#     time.sleep(1)
#
#     model = torch.load(model_path, weights_only=False)
#     model.to(device)
#     model.eval()
#
#     model_size_ram = asizeof.asizeof(model) / (1024 * 1)
#
#     batch = next(iter(DataLoader(test_data, batch_size=config["batch_size"])))
#     xb = batch[0].to(device)
#     cb = batch[1].to(device) if model_class.__name__ == "ConditionalVariationalAutoEncoder" else None
#
#     peak_ram = 0.0
#     if torch.cuda.is_available():
#         torch.cuda.reset_peak_memory_stats()
#         with torch.no_grad():
#             _ = model(xb, cb) if cb is not None else model(xb)
#         peak_ram = torch.cuda.max_memory_allocated() / (1024 ** 2)
#     else:
#         peak_usages = memory_usage((run_inference, (model, xb, cb)), interval=0.1, timeout=None)
#         peak_ram = max(peak_usages)  # In MB
#
#     Report.blue(f"📈 Peak memory during inference: {peak_ram:.2f} MB")
#
#     model_size_disk = os.path.getsize(model_path) / 1024 if model_path else 0
#     n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     test_metrics = compute_test_metrics(model, x_val, y_val, x_test, y_test, c_val, c_test, device)
#
#     return convert_to_serializable({
#         "Model": model_class.__name__,
#         "Train": not load_available,
#         "Params": n_params,
#         "TrainRAM_KB": round(train_ram, 2),
#         "InferencePeakRAM_MB": round(peak_ram, 2),
#         "ModelRAM_KB": round(model_size_ram, 2),
#         "TrainTime_second": round(t_train, 2),
#         "InferenceTime_ms": round(1000 * t_train, 4),
#         "DiskSize_KB": round(model_size_disk, 2),
#         "Val_MSE": round(val_err, 6),
#         **{k: round(v, 4) for k, v in test_metrics.items()}
#     })
#
#     # Use actual saved model size (not temp)
#     model_size = os.path.getsize(model_path) / 1024
#     n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     test_metrics = compute_test_metrics(model, x_val, y_val, x_test, y_test, c_val, c_test, device)
#
#     return convert_to_serializable({
#         "Model": model_class.__name__,
#         "Train": (not load_available),
#         "Params": n_params,
#         "TrainRAM_KB": round(train_ram, 2),
#         "InferenceRAM_KB": round(inf_ram, 2),
#         "InferenceAndModelRAM_KB": round(inf_ram + model_size_ram, 2),
#         "TrainTime_second": round(t_train, 2),
#         "InferenceTime_ms": round(t_inf * 1000, 4),
#         "CPU%": round(inf_cpu, 2),
#         "DiskSize_KB": round(model_size, 2),
#         "Val_MSE": round(val_err, 6),
#         **{k: round(v, 4) for k, v in test_metrics.items()}
#     })
#
#
# if __name__ == "__main__":
#
#     model_class = LSTMAutoEncoder
#
#     # initial configurations
#     BEST_CONFIG = {
#         "hidden_layers": [64, 32],
#         "latent_dim": 16,
#         "dropout": 0.1,
#         "activation": "relu",
#         "batch_size": 32,
#         "lr": 1e-3,
#         "epochs": 50,
#         "optimizer": "adam",
#         "window_size": 10
#     } if model_class.__name__ != "LSTMAutoEncoder" else {
#             "hidden_dim":  64,
#             "latent_dim": 16,
#             "num_layers": 2,
#             "batch_size": 32,
#             "lr": 1e-3,
#             "optimizer": "adam",
#             "epochs": 50,
#             "window_size": 10,
#         }
#
#     # prepare data
#     window_size = BEST_CONFIG.get("window_size") if model_class.__name__ == "LSTMAutoEncoder" else None
#     dataset_path = path.join('..', '..', 'input', 'Dataset.csv')
#     train_final, val_final, test_final, y_val, y_test = read_data(dataset_path, model_class, window_size)
#
#     # run the profiler
#     result = profile_model(model_class, BEST_CONFIG, train_final, val_final, test_final, y_val, y_test)
#
#     # save the results
#     output_csv_path = "footprint_results.csv"
#     if os.path.exists(output_csv_path):
#         existing_df = pd.read_csv(output_csv_path)
#         df = pd.concat([existing_df, pd.DataFrame([result])], ignore_index=True)
#     else:
#         df = pd.DataFrame([result])
#     df.to_csv(output_csv_path, index=False)
#     print(f"Results saved to {output_csv_path}")
#
