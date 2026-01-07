"""Interactive visualization of scribble generation using marimo.

This notebook allows you to:
- Create sample masks
- Visualize generated scribbles
- Tune generation parameters
- Compare different configurations
"""

import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from mask_to_scribble import RangeF, RangeI, ScribbleConfig, ScribbleGenerator

    return RangeF, RangeI, ScribbleConfig, ScribbleGenerator, mo, np, plt


@app.cell
def _(mo):
    mo.md("""
    # Scribble Generation Visualization

    This notebook demonstrates the mask-to-scribble generation algorithm.

    Generate human-like scribble annotations from binary defect masks for
    Positive-Unlabeled anomaly detection tasks.
    """)


@app.cell
def _(mo):
    mo.md("""
    ## 1. Create Sample Masks
    """)


@app.cell
def _(mo):
    mask_type = mo.ui.dropdown(
        options={
            "square": "square",
            "circle": "circle",
            "ellipse": "ellipse",
            "irregular": "irregular",
            "small": "small",
            "large": "large",
        },
        value="square",
        label="Mask Type",
    )
    mask_type
    return (mask_type,)


@app.cell
def _(mask_type, np):
    def create_sample_mask(mask_type_value: str, size: int = 256) -> np.ndarray:
        """Create various sample masks for testing."""
        mask = np.zeros((size, size), dtype=np.uint8)

        if mask_type_value == "square":
            mask[80:180, 80:180] = 255
        elif mask_type_value == "circle":
            y, x = np.ogrid[:size, :size]
            center_y, center_x = size // 2, size // 2
            circle_mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= 60**2
            mask[circle_mask] = 255
        elif mask_type_value == "ellipse":
            y, x = np.ogrid[:size, :size]
            center_y, center_x = size // 2, size // 2
            ellipse_mask = ((x - center_x) / 80) ** 2 + ((y - center_y) / 40) ** 2 <= 1
            mask[ellipse_mask] = 255
        elif mask_type_value == "irregular":
            # Create an irregular shape
            mask[60:120, 80:180] = 255
            mask[100:180, 100:140] = 255
            mask[140:200, 60:100] = 255
        elif mask_type_value == "small":
            mask[110:130, 110:130] = 255
        elif mask_type_value == "large":
            mask[30:230, 40:220] = 255
        else:
            msg = f"`{mask_type_value}` is invalid."
            raise ValueError(msg)

        return mask

    sample_mask = create_sample_mask(mask_type.value)
    return create_sample_mask, sample_mask


@app.cell
def _(plt, sample_mask):
    plt.imshow(sample_mask, cmap="gray", vmin=0, vmax=255)


@app.cell
def _(mo):
    mo.md("""
    ## 2. Configure Scribble Generation
    """)


@app.cell
def _(mo):
    mo.md("## 生成パラメータ（Multi-scribble）")

    # --- core multi-scribble controls ---
    num_scribbles = mo.ui.slider(1, 6, value=3, step=1, label="K（ストローク本数）")
    coverage_cap = mo.ui.slider(
        0.05, 0.40, value=0.15, step=0.01, label="Coverage cap（面に寄りすぎ防止）",
    )
    coverage_check_each_stroke = mo.ui.checkbox(
        label="各ストローク追加ごとにcoverageチェック", value=True,
    )

    # --- points: base & variation ---
    use_area_based = mo.ui.checkbox(
        label="点数Nを面積依存にする（fixed_num_pointsを無視）", value=False,
    )
    fixed_num_points = mo.ui.slider(2, 12, value=5, step=1, label="固定N点（面積依存OFF時）")

    # Variation mode: scalar only or per-stroke range
    var_mode = mo.ui.dropdown(
        label="ストローク間の変動モード",
        options=["scalar（全ストローク同一）", "range（ストロークごとに決定論サンプル）"],
        value="range（ストロークごとに決定論サンプル）",
    )

    # --- center/boundary mixture ---
    mix_center_base = mo.ui.slider(
        0.0, 1.0, value=0.45, step=0.01, label="mix_center（中心寄り混合率）ベース",
    )
    mix_center_var = mo.ui.slider(0.0, 0.50, value=0.20, step=0.01, label="mix_center 変動幅（±）")

    center_sigma_scale_base = mo.ui.slider(
        0.05, 0.80, value=0.25, step=0.01, label="center_sigma_scale ベース",
    )
    center_sigma_scale_var = mo.ui.slider(
        0.0, 0.40, value=0.10, step=0.01, label="center_sigma_scale 変動幅（±）",
    )

    boundary_sigma_base = mo.ui.slider(
        1.0, 20.0, value=6.0, step=0.5, label="boundary_sigma（px）ベース",
    )
    boundary_sigma_var = mo.ui.slider(
        0.0, 10.0, value=2.0, step=0.5, label="boundary_sigma 変動幅（±）",
    )

    # --- sampling separation ---
    min_dist_scale_base = mo.ui.slider(
        0.02, 0.30, value=0.10, step=0.01, label="min_dist_scale ベース",
    )
    min_dist_scale_var = mo.ui.slider(
        0.0, 0.15, value=0.03, step=0.01, label="min_dist_scale 変動幅（±）",
    )

    # --- routing ---
    jitter_strength_base = mo.ui.slider(
        0.0, 0.60, value=0.25, step=0.01, label="jitter_strength ベース",
    )
    jitter_strength_var = mo.ui.slider(
        0.0, 0.60, value=0.10, step=0.01, label="jitter_strength 変動幅（±）",
    )

    jitter_smooth_ksize_base = mo.ui.slider(
        1, 51, value=9, step=2, label="jitter_smooth_ksize（奇数）ベース",
    )
    jitter_smooth_ksize_var = mo.ui.slider(
        0, 20, value=6, step=2, label="jitter_smooth_ksize 変動幅（±, 偶数step）",
    )

    center_bias_base = mo.ui.slider(0.0, 0.40, value=0.10, step=0.01, label="center_bias ベース")
    center_bias_var = mo.ui.slider(
        0.0, 0.40, value=0.05, step=0.01, label="center_bias 変動幅（±）",
    )

    # --- thickness ---
    min_radius = mo.ui.slider(1, 15, value=2, step=1, label="min_radius（最小線半径）")
    radius_scale = mo.ui.slider(
        0.0, 0.05, value=0.012, step=0.001, label="radius_scale（sqrt(area)に掛ける）",
    )
    max_radius = mo.ui.slider(1, 50, value=20, step=1, label="max_radius（最大線半径）")

    # --- soft labels ---
    use_soft = mo.ui.checkbox(label="ソフトラベル出力（0..1）", value=False)
    soft_sigma = mo.ui.slider(0.5, 20.0, value=2.0, step=0.5, label="soft_label_sigma")

    # --- debug ---
    show_debug = mo.ui.checkbox(label="デバッグ（選択点とMSTエッジを可視化）", value=True)
    return (
        boundary_sigma_base,
        boundary_sigma_var,
        center_bias_base,
        center_bias_var,
        center_sigma_scale_base,
        center_sigma_scale_var,
        coverage_cap,
        coverage_check_each_stroke,
        fixed_num_points,
        jitter_smooth_ksize_base,
        jitter_smooth_ksize_var,
        jitter_strength_base,
        jitter_strength_var,
        max_radius,
        min_dist_scale_base,
        min_dist_scale_var,
        min_radius,
        mix_center_base,
        mix_center_var,
        num_scribbles,
        radius_scale,
        show_debug,
        soft_sigma,
        use_area_based,
        use_soft,
        var_mode,
    )


@app.cell
def _(
    boundary_sigma_base,
    boundary_sigma_var,
    center_bias_base,
    center_bias_var,
    center_sigma_scale_base,
    center_sigma_scale_var,
    coverage_cap,
    coverage_check_each_stroke,
    fixed_num_points,
    jitter_smooth_ksize_base,
    jitter_smooth_ksize_var,
    jitter_strength_base,
    jitter_strength_var,
    max_radius,
    min_dist_scale_base,
    min_dist_scale_var,
    min_radius,
    mix_center_base,
    mix_center_var,
    mo,
    num_scribbles,
    radius_scale,
    show_debug,
    soft_sigma,
    use_area_based,
    use_soft,
    var_mode,
):
    mo.vstack(
        [
            mo.hstack([num_scribbles, coverage_cap, coverage_check_each_stroke]),
            mo.hstack([use_area_based, fixed_num_points, var_mode, show_debug]),
            mo.md("### 点選択（中心/境界）"),
            mo.hstack([mix_center_base, mix_center_var]),
            mo.hstack([center_sigma_scale_base, center_sigma_scale_var]),
            mo.hstack([boundary_sigma_base, boundary_sigma_var]),
            mo.md("### 点の分散"),
            mo.hstack([min_dist_scale_base, min_dist_scale_var]),
            mo.md("### 経路（ジッタ・中心寄せ）"),
            mo.hstack([jitter_strength_base, jitter_strength_var]),
            mo.hstack([jitter_smooth_ksize_base, jitter_smooth_ksize_var]),
            mo.hstack([center_bias_base, center_bias_var]),
            mo.md("### 太さ / ラベル"),
            mo.hstack([min_radius, radius_scale, max_radius]),
            mo.hstack([use_soft, soft_sigma]),
        ],
    )


@app.cell(hide_code=True)
def _(
    RangeF,
    RangeI,
    ScribbleConfig,
    ScribbleGenerator,
    boundary_sigma_base,
    boundary_sigma_var,
    center_bias_base,
    center_bias_var,
    center_sigma_scale_base,
    center_sigma_scale_var,
    coverage_cap,
    coverage_check_each_stroke,
    fixed_num_points,
    jitter_smooth_ksize_base,
    jitter_smooth_ksize_var,
    jitter_strength_base,
    jitter_strength_var,
    max_radius,
    min_dist_scale_base,
    min_dist_scale_var,
    min_radius,
    mix_center_base,
    mix_center_var,
    num_scribbles,
    radius_scale,
    sample_mask,
    soft_sigma,
    use_area_based,
    use_soft,
    var_mode,
):
    def _rangef(base: float, var: float, lo: float | None = None, hi: float | None = None):
        # RangeF はあなたのA実装にある想定
        a = base - var
        b = base + var
        if lo is not None:
            a = max(a, lo)
        if hi is not None:
            b = min(b, hi)
        # a <= b を保証
        if b < a:
            a, b = b, a
        return RangeF(a, b)

    def _rangei_odd(base: int, var: int, lo: int, hi: int):
        # odd を維持したいので、まず範囲を作っておいて内部で奇数に寄せる運用でもOK。
        # ここでは「範囲は整数」で作るだけにして、A実装側で奇数化はしない前提。
        a = max(lo, base - var)
        b = min(hi, base + var)
        if b < a:
            a, b = b, a
        return RangeI(a, b)

    is_range = var_mode.value.startswith("range")

    mix_center_spec = (
        _rangef(float(mix_center_base.value), float(mix_center_var.value), lo=0.0, hi=1.0)
        if is_range
        else float(mix_center_base.value)
    )
    center_sigma_scale_spec = (
        _rangef(float(center_sigma_scale_base.value), float(center_sigma_scale_var.value), lo=0.01)
        if is_range
        else float(center_sigma_scale_base.value)
    )
    boundary_sigma_spec = (
        _rangef(float(boundary_sigma_base.value), float(boundary_sigma_var.value), lo=0.5)
        if is_range
        else float(boundary_sigma_base.value)
    )
    min_dist_scale_spec = (
        _rangef(float(min_dist_scale_base.value), float(min_dist_scale_var.value), lo=0.0)
        if is_range
        else float(min_dist_scale_base.value)
    )

    jitter_strength_spec = (
        _rangef(float(jitter_strength_base.value), float(jitter_strength_var.value), lo=0.0)
        if is_range
        else float(jitter_strength_base.value)
    )
    center_bias_spec = (
        _rangef(float(center_bias_base.value), float(center_bias_var.value), lo=0.0)
        if is_range
        else float(center_bias_base.value)
    )

    # IntSpec for ksize
    ksize_base = int(jitter_smooth_ksize_base.value)
    ksize_var = int(jitter_smooth_ksize_var.value)
    jitter_ksize_spec = (
        _rangei_odd(ksize_base, ksize_var, lo=1, hi=51)
        if is_range
        else int(jitter_smooth_ksize_base.value)
    )

    cfg = ScribbleConfig(
        # multi-scribble
        num_scribbles=int(num_scribbles.value),
        coverage_cap=float(coverage_cap.value),
        coverage_check_each_stroke=bool(coverage_check_each_stroke.value),
        # points count
        fixed_num_points=None if use_area_based.value else int(fixed_num_points.value),
        min_points=3,
        max_points=7,
        # per-stroke varying specs
        mix_center=mix_center_spec,
        center_sigma_scale=center_sigma_scale_spec,
        boundary_sigma=boundary_sigma_spec,
        min_dist_scale=min_dist_scale_spec,
        jitter_strength=jitter_strength_spec,
        jitter_smooth_ksize=jitter_ksize_spec,
        center_bias=center_bias_spec,
        # thickness
        min_radius=int(min_radius.value),
        radius_scale=float(radius_scale.value),
        max_radius=int(max_radius.value),
        # soft labels
        use_soft_labels=bool(use_soft.value),
        soft_label_sigma=float(soft_sigma.value),
    )

    generator = ScribbleGenerator(cfg)
    generated_scribble = generator.from_mask(sample_mask)

    return generated_scribble, generator


@app.cell
def _(mo):
    mo.md("""
    ## 3. Visualization
    """)


@app.cell
def _(generated_scribble, plt, sample_mask, use_soft):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(sample_mask, cmap="gray")
    axes[0].set_title("Original Mask")
    axes[0].axis("off")

    if use_soft.value:
        im = axes[1].imshow(generated_scribble, cmap="hot", vmin=0, vmax=1)
        axes[1].set_title("Generated Scribble (Soft Labels)")
        fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    else:
        axes[1].imshow(generated_scribble, cmap="gray")
        axes[1].set_title("Generated Scribble")
    axes[1].axis("off")

    axes[2].imshow(sample_mask, cmap="gray", alpha=0.5)
    if use_soft.value:
        axes[2].imshow(generated_scribble, cmap="hot", alpha=0.8, vmin=0, vmax=1)
    else:
        axes[2].imshow(generated_scribble, cmap="Reds", alpha=0.8)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    plt.tight_layout()

    fig


@app.cell
def _(generated_scribble, mo, np, sample_mask):
    mask_area = np.sum(sample_mask > 0)
    scribble_area = (
        np.sum(generated_scribble > 0)
        if generated_scribble.dtype == np.uint8
        else np.sum(generated_scribble > 0.5)
    )
    coverage = (scribble_area / mask_area * 100) if mask_area > 0 else 0

    mo.md(f"""
    ### Statistics

    - **Mask Area**: {mask_area:,} pixels
    - **Scribble Area**: {scribble_area:,} pixels
    - **Coverage**: {coverage:.2f}% of mask
    """)


@app.cell
def _(mo):
    mo.md("""
    ## 4. Determinism Check

    Verify that the same mask always produces the same scribble (reproducibility).
    """)


@app.cell
def _(generator, np, sample_mask):
    # Generate multiple times
    scribble_1 = generator.from_mask(sample_mask)
    scribble_2 = generator.from_mask(sample_mask)
    scribble_3 = generator.from_mask(sample_mask)

    is_deterministic = np.array_equal(scribble_1, scribble_2) and np.array_equal(
        scribble_2,
        scribble_3,
    )
    return (is_deterministic,)


@app.cell
def _(is_deterministic, mo):
    if is_deterministic:
        mo.callout(
            "✅ Determinism verified! Same mask produces identical scribbles.",
            kind="success",
        )
    else:
        mo.callout(
            "⚠️ Warning: Non-deterministic behavior detected!",
            kind="danger",
        )


@app.cell
def _(mo):
    mo.md("""
    ## 5. Multiple Examples

    Compare different mask types side-by-side.
    """)


@app.cell
def _(create_sample_mask, generator, mo, plt):
    mask_types = ["square", "circle", "ellipse", "irregular", "small", "large"]

    _fig, _axes = plt.subplots(2, 6, figsize=(18, 6))

    for idx, _mask_type in enumerate(mask_types):
        _mask = create_sample_mask(_mask_type)
        _scribble = generator.from_mask(_mask)

        # Show mask
        _axes[0, idx].imshow(_mask, cmap="gray")
        _axes[0, idx].set_title(_mask_type.capitalize())
        _axes[0, idx].axis("off")

        # Show scribble
        _axes[1, idx].imshow(_scribble, cmap="gray")
        _axes[1, idx].axis("off")

    _axes[0, 0].set_ylabel("Mask", fontsize=12)
    _axes[1, 0].set_ylabel("Scribble", fontsize=12)

    plt.tight_layout()
    mo.ui.plotly(_fig)


@app.cell
def _(mo):
    mo.md("""
    ---

    ## About

    This visualization demonstrates the **mask-to-scribble** package:

    - **Deterministic**: Same mask always produces same scribble
    - **Human-like**: Combines skeleton + contour strokes
    - **Adaptive**: Thickness scales with defect size
    - **Minimal**: Sparse annotation (not full mask)

    Designed for Positive-Unlabeled anomaly detection (PU-SAC, Dinomaly).
    """)


if __name__ == "__main__":
    app.run()
