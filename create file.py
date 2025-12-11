import numpy as np
import tifffile

# Dimensions
WIDTH, HEIGHT = 6000, 4000

# Optionnel : seed pour reproductibilité
# np.random.seed(42)


def normalize_to_uint16(img_float):
    """
    Normalise un tableau float quelconque en [0, 65535] et renvoie un uint16.
    img_float est supposé de shape (H, W, 3) ou (H, W).
    """
    img_min = img_float.min()
    img_max = img_float.max()
    # On évite une division par zéro
    denom = img_max - img_min if img_max != img_min else 1.0
    img_norm = (img_float - img_min) / denom
    img_uint16 = (img_norm * 65535.0).astype(np.uint16)
    return img_uint16


def main():
    # --- 1. Bruit blanc pur (de référence) ---
    # Bruit uniforme i.i.d. dans [0, 1)
    noise_pure = np.random.rand(HEIGHT, WIDTH, 3).astype(np.float32)
    noise_pure_u16 = (noise_pure * 65535.0).astype(np.uint16)
    tifffile.imwrite("noise_pur.tiff", noise_pure_u16)
    print("Écrit : noise_pur.tiff")

    # Pour les corrélations, on part de la même base float
    base = noise_pure

    # --- 2. Corrélation horizontale (colonnes) ---
    # Chaque pixel ≈ moyenne de lui-même + voisin gauche + voisin droit
    noise_horiz = (
        base
        + np.roll(base, shift=1, axis=1)   # décalage vers la droite
        + np.roll(base, shift=-1, axis=1)  # décalage vers la gauche
    ) / 3.0
    noise_horiz_u16 = normalize_to_uint16(noise_horiz)
    tifffile.imwrite("noise_corr_horiz.tiff", noise_horiz_u16)
    print("Écrit : noise_corr_horiz.tiff")

    # --- 3. Corrélation verticale (lignes) ---
    # Chaque pixel ≈ moyenne de lui-même + voisin haut + voisin bas
    noise_vert = (
        base
        + np.roll(base, shift=1, axis=0)   # décalage vers le bas
        + np.roll(base, shift=-1, axis=0)  # décalage vers le haut
    ) / 3.0
    noise_vert_u16 = normalize_to_uint16(noise_vert)
    tifffile.imwrite("noise_corr_vert.tiff", noise_vert_u16)
    print("Écrit : noise_corr_vert.tiff")

    # --- 4. Corrélation isotrope (flou 2D simple) ---
    # Chaque pixel ≈ moyenne de lui-même + voisins haut/bas/gauche/droite
    noise_iso = (
        base
        + np.roll(base, shift=1, axis=0)    # bas
        + np.roll(base, shift=-1, axis=0)   # haut
        + np.roll(base, shift=1, axis=1)    # droite
        + np.roll(base, shift=-1, axis=1)   # gauche
    ) / 5.0
    noise_iso_u16 = normalize_to_uint16(noise_iso)
    tifffile.imwrite("noise_corr_iso.tiff", noise_iso_u16)
    print("Écrit : noise_corr_iso.tiff")


if __name__ == "__main__":
    main()
