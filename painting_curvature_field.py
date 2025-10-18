# ============================================================
# Painting Curvature Field Analysis
# ============================================================
# This script computes and visualizes the curvature field in paintings.
# It provides both standalone curvature maps and overlays on the original artwork.
# Intended for research on visual geometry and mathematical aesthetics.
# ============================================================

# ---------------------
# 1) Imports & Setup
# ---------------------
import os
import cv2
import warnings
import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

FIG_DIR = "./figures"
os.makedirs(FIG_DIR, exist_ok=True)

TARGET_SIZE = 512  # px
DPI_FIG = 300      # for publication-quality output

# Path to painting
PAINTINGS = {
    "The Starry Night (June 1889)": "./the_starry_night.png"
}

# ============================================================
# 2) Compute curvature field
# ============================================================
def compute_curvature_field(image_path: str, target_size: int = 512) -> np.ndarray:
    """Compute the 2D curvature field from a grayscale luminance map of the painting."""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(cv2.resize(img, (target_size, target_size)), cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    norm = gray / 255.0

    # Image gradients
    Gx = cv2.Sobel(norm, cv2.CV_64F, 1, 0, ksize=3)
    Gy = cv2.Sobel(norm, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(Gx**2 + Gy**2) + 1e-6

    # Normalized gradients
    Nx, Ny = Gx / mag, Gy / mag
    dNx_dx = cv2.Sobel(Nx, cv2.CV_64F, 1, 0, ksize=3)
    dNy_dy = cv2.Sobel(Ny, cv2.CV_64F, 0, 1, ksize=3)

    # Curvature field (divergence of the unit normal)
    curvature = dNx_dx + dNy_dy
    return curvature

# ============================================================
# 3) Plot curvature field
# ============================================================
def plot_curvature_field(title: str, curvature: np.ndarray, cmap: str = "coolwarm"):
    """Plot the curvature field as a heatmap."""
    plt.figure(figsize=(6, 6))
    plt.imshow(curvature, cmap=cmap)
    plt.axis("off")
    plt.title(f"{title}\nCurvature Field", fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/{title.replace(' ', '_')}_curvature.png", dpi=DPI_FIG)
    plt.show()

# ============================================================
# 4) Overlay curvature on painting
# ============================================================
def overlay_curvature_on_painting(title: str, image_path: str, curvature: np.ndarray):
    """Overlay the curvature field over the original painting."""
    img_rgb = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (curvature.shape[1], curvature.shape[0]))
    img_norm = img_rgb / 255.0

    plt.figure(figsize=(7, 7))
    plt.imshow(img_norm, alpha=0.55)
    im = plt.imshow(curvature, cmap="jet", alpha=0.75)
    plt.axis("off")

    cbar = plt.colorbar(im, orientation="horizontal", pad=0.03, fraction=0.046)
    cbar.set_label(r"Curvature $|\kappa|$", fontsize=14)
    cbar.ax.tick_params(labelsize=12, length=0)
    cbar.formatter.set_powerlimits((-2, 2))
    cbar.update_ticks()

    plt.tight_layout()
    plt.savefig(
        f"{FIG_DIR}/{title.replace(' ', '_')}_curvature_overlay.png",
        dpi=DPI_FIG,
        bbox_inches="tight",
        pad_inches=0.05,
    )
    plt.show()

# ============================================================
# 5) Run analysis
# ============================================================
if __name__ == "__main__":
    for title, path in PAINTINGS.items():
        curvature = compute_curvature_field(path)
        plot_curvature_field(title, curvature)
        overlay_curvature_on_painting(title, path, curvature)
