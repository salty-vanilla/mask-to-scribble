import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from PIL import Image

    from mask_to_scribble import ScribbleConfig, ScribbleGenerator
    return Image, Path, ScribbleConfig, ScribbleGenerator, np, plt


@app.cell
def _(Path):
    dataset_dir = Path("/Users/van/Downloads/mvtec_anomaly_detection")
    mask_dir = dataset_dir / "bottle" / "ground_truth" / "broken_large"
    return (mask_dir,)


@app.cell
def _(mask_dir):
    mask_paths = list(mask_dir.glob("*.png"))
    mask_paths
    return (mask_paths,)


@app.cell
def _(ScribbleConfig, ScribbleGenerator):
    cfg = ScribbleConfig(num_scribbles=2)
    generator = ScribbleGenerator(cfg)
    return (generator,)


@app.cell
def _(Image, generator, mask_paths, np, plt):
    for mask_path in mask_paths[:3]:
        mask = np.array(Image.open(mask_path))
        scribble = generator.from_mask(mask)

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title("Mask")
        plt.imshow(mask, cmap="gray")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.title("Scribble")
        plt.imshow(scribble, cmap="gray")
        plt.axis("off")

        plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
