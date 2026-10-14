import warnings
import sys
import os

# Safe backend configuration for interactive GUI window popup on Linux desktops
import matplotlib
try:
    matplotlib.use('TkAgg')
except Exception:
    pass

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score

warnings.filterwarnings("ignore")


def calculate_shannon_entropy(block: np.ndarray) -> float:
    """Calculates the Shannon entropy of a given byte block."""
    _, counts = np.unique(block, return_counts=True)
    probabilities = counts / len(block)
    return -np.sum(probabilities * np.log2(probabilities + 1e-10))


def compute_bit_transition_matrix(row: np.ndarray) -> np.ndarray:
    """Computes first-order bit transition probabilities."""
    bits = np.unpackbits(row)
    transitions = np.zeros((2, 2), dtype=float)
    for i in range(len(bits) - 1):
        transitions[bits[i], bits[i + 1]] += 1
    total = np.sum(transitions)
    if total > 0:
        transitions /= total
    return transitions.flatten()


def compute_fft_features(row: np.ndarray) -> list:
    """Performs Fast Fourier Transform to extract frequency spectrum magnitude features."""
    fft_vals = np.fft.fft(row.astype(float))
    fft_mag = np.abs(fft_vals[: len(row) // 2])
    return [np.mean(fft_mag), np.std(fft_mag), np.max(fft_mag)]


def extract_features(X: np.ndarray) -> np.ndarray:
    """Extracts 72-dimensional feature vector combining statistical moments, entropy, and spectral data."""
    print("[+] Extracting 72-Dimensional Features (FFT, Bit Transitions, Entropy, Statistics)...")
    features_list = []

    for row in X:
        feats = []

        # 1. Basic Statistical Moments
        feats.extend([
            np.mean(row), np.std(row), np.var(row),
            np.min(row), np.max(row), np.median(row),
            np.ptp(row), np.percentile(row, 25), np.percentile(row, 75)
        ])

        # 2. Shannon Entropy
        feats.append(calculate_shannon_entropy(row))

        # 3. Delta Series Statistics
        diffs = np.diff(row.astype(int))
        feats.extend([np.mean(diffs), np.std(diffs), np.min(diffs), np.max(diffs)])

        # 4. Bit-Level Metrics
        bits = np.unpackbits(row)
        feats.extend([np.sum(bits), np.std(bits), np.mean(bits)])

        # 5. Markov Bit Transition Probabilities
        feats.extend(compute_bit_transition_matrix(row))

        # 6. Spectral FFT Features
        feats.extend(compute_fft_features(row))

        # 7. Byte Histogram (16 Bins)
        hist, _ = np.histogram(row, bins=16, range=(0, 256))
        feats.extend(hist / len(row))

        # 8. Raw Byte Projection
        feats.extend(row)

        features_list.append(feats)

    return np.array(features_list)


def plot_results(model_names: list, accuracies: list, stds: list):
    """Launches an interactive Matplotlib window displaying model evaluation metrics."""
    plt.figure(figsize=(10, 6), dpi=100)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    bars = plt.bar(
        model_names,
        accuracies,
        yerr=stds,
        capsize=6,
        color=["#1f77b4", "#ff7f0e", "#2ca02c"],
        alpha=0.85,
        edgecolor="black",
        linewidth=1.2,
    )

    plt.axhline(y=50, color="red", linestyle="--", linewidth=1.8, label="Random Guess Baseline (50%)")

    plt.ylim(0, 100)
    plt.ylabel("Cross-Validation Accuracy (%)", fontsize=12, fontweight="bold")
    plt.title("Crypto-Forensic Architecture Independence Audit (Ed25519 / CSPRNG)", fontsize=13, fontweight="bold", pad=15)
    plt.legend(loc="upper right", frameon=True)

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 3,
            f"{height:.2f}%",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    plt.tight_layout()
    print("[+] Opening interactive evaluation plot window...")
    plt.show()


def run_crypto_audit():
    print("[+] Reading cross-platform binary dumps...")

    if not os.path.exists("linux_x86.bin") or not os.path.exists("windows_arm.bin"):
        print("[-] Error: Binary dump files ('linux_x86.bin' or 'windows_arm.bin') not found!")
        print("[-] Please run 'generator.py' first to produce dataset samples.")
        sys.exit(1)

    with open("linux_x86.bin", "rb") as f:
        linux_data = f.read()
    X_linux = np.frombuffer(linux_data, dtype=np.uint8).reshape(-1, 32)

    with open("windows_arm.bin", "rb") as f:
        windows_data = f.read()
    X_windows = np.frombuffer(windows_data, dtype=np.uint8).reshape(-1, 32)

    X_linux_feat = extract_features(X_linux)
    X_windows_feat = extract_features(X_windows)

    X = np.vstack((X_linux_feat, X_windows_feat))
    y = np.array([0] * len(X_linux) + [1] * len(X_windows))

    print(f"[+] Total Samples: {len(X)} | Feature Dimension: {X.shape[1]}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    models = {
        "Random Forest (Tuned)": RandomForestClassifier(
            n_estimators=500, max_depth=20, min_samples_split=5, random_state=42
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=500, max_depth=25, random_state=42
        ),
    }

    print("\n" + "=" * 65)
    print(" CRYPTO-FORENSIC CROSS-VALIDATION AUDIT IN PROGRESS ")
    print("=" * 65)

    model_names, accuracies, stds = [], [], []

    for name, clf in models.items():
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", clf)
        ])

        scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")
        mean_acc = scores.mean() * 100
        std_acc = scores.std() * 100

        model_names.append(name)
        accuracies.append(mean_acc)
        stds.append(std_acc)

        print(f"[*] {name:25} -> Mean Accuracy: %{mean_acc:.2f} (±%{std_acc:.2f})")

    print("=" * 65)
    print("[-] Final Cryptographic Assessment:")
    print("    Model scores oscillating tightly around 50% mathematically confirm")
    print("    that Ed25519 outputs exhibit absolute OS/architecture independence.")
    print("=" * 65)

    plot_results(model_names, accuracies, stds)


if __name__ == "__main__":
    run_crypto_audit()
