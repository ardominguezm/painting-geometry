# ============================================================
# Flow & Curvature Field Analysis in Van Gogh's "The Starry Night"
# ============================================================
# Author: Andy Domínguez
# Description:
#   Computes and visualizes the 2D curvature field of a painting
#   using the luminance gradient method. The analysis is applied
#   directly on the high-resolution grayscale image (no resizing).
# ============================================================

# -------------------------
# 1) Imports & Configuration
# -------------------------
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# Output directory for figures
FIG_DIR = "./figures"
os.makedirs(FIG_DIR, exist_ok=True)

# Path to the high-resolution painting
PAINTINGS = {
    "The Starry Night (June 1889)": "./2728px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg"
}

# Figure quality
DPI_FIG = 300  # publication quality


# -------------------------
# 2) Image Preprocessing
# -------------------------
def preprocess_image(image_path: str):
    """
    Load and preprocess the painting:
    - Convert to RGB
    - Convert to grayscale
    - Apply Gaussian blur
    - Normalize luminance to [0, 1]
    """
    img_bgr = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)
    norm = gray_blur / 255.0
    return img_rgb, gray, norm


def plot_preprocessing_steps(title: str, img_rgb: np.ndarray, gray: np.ndarray, norm: np.ndarray):
    """Visualize the preprocessing pipeline."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    steps = [
        ("Original RGB", img_rgb, None),
        ("Grayscale", gray, "gray"),
        ("Normalized [0,1]", norm, "gray")
    ]
    for ax, (label, img, cmap) in zip(axes, steps):
        ax.imshow(img, cmap=cmap)
        ax.set_title(label, fontsize=12)
        ax.axis("off")

    plt.suptitle(f"{title} – Preprocessing Steps", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/{title.replace(' ', '_')}_preprocessing.png", dpi=DPI_FIG)
    plt.show()


# -------------------------
# 3) Curvature Field Computation
# -------------------------
def compute_curvature_field(norm: np.ndarray) -> np.ndarray:
    """
    Compute the 2D curvature field κ from the normalized luminance.
    The curvature is defined as the divergence of the unit normal vector field:
        κ = div(N) = ∂Nx/∂x + ∂Ny/∂y
    where N = (∇I) / |∇I|.
    """
    Gx = cv2.Sobel(norm, cv2.CV_64F, 1, 0, ksize=3)
    Gy = cv2.Sobel(norm, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(Gx**2 + Gy**2) + 1e-6  # gradient magnitude
    Nx, Ny = Gx / mag, Gy / mag
    dNx_dx = cv2.Sobel(Nx, cv2.CV_64F, 1, 0, ksize=3)
    dNy_dy = cv2.Sobel(Ny, cv2.CV_64F, 0, 1, ksize=3)
    curvature = dNx_dx + dNy_dy
    return curvature


# -------------------------
# 4) Visualization
# -------------------------
def plot_curvature_field(title: str, curvature: np.ndarray, cmap: str = "coolwarm"):
    """Display the curvature field as a heatmap."""
    plt.figure(figsize=(8, 8))
    plt.imshow(curvature, cmap=cmap)
    plt.axis("off")
    plt.title(f"{title}\nCurvature Field (κ)", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/{title.replace(' ', '_')}_curvature.png", dpi=DPI_FIG)
    plt.show()


def overlay_curvature_on_painting(title: str, img_rgb: np.ndarray, curvature: np.ndarray):
    """Overlay curvature field onto the original high-resolution painting."""
    img_norm = img_rgb / 255.0
    plt.figure(figsize=(10, 10))
    plt.imshow(img_norm, alpha=0.55)
    im = plt.imshow(curvature, cmap="jet", alpha=0.75)
    plt.axis("off")

    cbar = plt.colorbar(im, orientation="horizontal", pad=0.03, fraction=0.046)
    cbar.set_label(r"Curvature $\kappa$", fontsize=18)
    cbar.ax.tick_params(labelsize=16, length=0)

    plt.tight_layout()
    plt.savefig(
        f"{FIG_DIR}/{title.replace(' ', '_')}_curvature_overlay.png",
        dpi=DPI_FIG,
        bbox_inches="tight",
        pad_inches=0.05,
    )
    plt.show()


# -------------------------
# 5) Main Execution
# -------------------------
if __name__ == "__main__":
    for title, path in PAINTINGS.items():
        img_rgb, gray, norm = preprocess_image(path)
        plot_preprocessing_steps(title, img_rgb, gray, norm)
        curvature = compute_curvature_field(norm)
        plot_curvature_field(title, curvature)
        overlay_curvature_on_painting(title, img_rgb, curvature)

