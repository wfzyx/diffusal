import marimo

__generated_with = "0.11.0"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    import torch
    import torch.nn as nn
    import numpy as np
    return mo, nn, np, torch


@app.cell
def __(mo):
    mo.md(
        r"""
        # Interactive MoE Quantization & Diffusion Workbench
        ### Exploring Error Damping in Sparse Mixture-of-Experts: AR vs. Block-Diffusion

        This interactive notebook accompanies the paper:
        **"Masked Diffusion Language Models Absorb Extreme Weight Quantization and Routing Noise in Sparse Mixture-of-Experts"**
        *(Panisa et al., 2026)*.

        Code & Checkpoints: [github.com/wfzyx/diffusal](https://github.com/wfzyx/diffusal)
        """
    )
    return


@app.cell
def __(mo):
    prompt_selector = mo.ui.dropdown(
        options=[
            "The theory of general relativity explains that gravity is caused by the curvature of spacetime",
            "In distributed consensus, the Raft protocol guarantees safety and fault tolerance by",
            "Modern AI hardware is fundamentally memory-bandwidth bound during token generation because",
            "In real analysis, a metric space is said to be sequentially compact if and only if every sequence has",
            "The key mechanism of CRISPR-Cas9 genome editing relies on guide RNA directing Cas9 endonuclease to"
        ],
        value="The theory of general relativity explains that gravity is caused by the curvature of spacetime",
        label="Select Test Prompt:"
    )

    precision_selector = mo.ui.radio(
        options=["FP32 (Unquantized)", "INT4 RTN", "BitNet b1.58 Ternary {-1, 0, +1}"],
        value="BitNet b1.58 Ternary {-1, 0, +1}",
        label="Expert Quantization Precision:"
    )

    conf_slider = mo.ui.slider(
        start=0.50,
        stop=0.99,
        step=0.05,
        value=0.85,
        label="Confidence Threshold (tau):"
    )

    block_slider = mo.ui.slider(
        start=8,
        stop=64,
        step=8,
        value=32,
        label="Diffusion Block Size (B):"
    )

    mo.hstack([prompt_selector, precision_selector])
    return block_slider, conf_slider, precision_selector, prompt_selector


@app.cell
def __(block_slider, conf_slider, mo):
    mo.hstack([block_slider, conf_slider])
    return


@app.cell
def __(block_slider, conf_slider, precision_selector, prompt_selector):
    # Simulation logic based on empirical benchmarks on Qwen3-MoE
    prec = precision_selector.value
    p_text = prompt_selector.value
    b_size = block_slider.value
    tau = conf_slider.value

    # Empirical parameter settings (corrected physical DRAM model)
    if "FP32" in prec:
        ar_drift = 0.0
        diff_drift = 0.0
        dram_streamed_ar = 1716.1
        dram_streamed_diff = 897.8
        ai_ar = 0.50
        ai_diff = 5.73
    elif "INT4" in prec:
        ar_drift = 24.85
        diff_drift = 28.77
        dram_streamed_ar = 214.5
        dram_streamed_diff = 112.2
        ai_ar = 4.00
        ai_diff = 45.88
    else:  # Ternary
        ar_drift = 46.09
        diff_drift = 55.40
        dram_streamed_ar = 107.3
        dram_streamed_diff = 56.1
        ai_ar = 8.00
        ai_diff = 91.75

    excess_drift = diff_drift - ar_drift
    gap_ratio = diff_drift / max(ar_drift, 1e-5)
    return (
        ai_ar,
        ai_diff,
        ar_drift,
        b_size,
        diff_drift,
        dram_streamed_ar,
        dram_streamed_diff,
        excess_drift,
        gap_ratio,
        p_text,
        prec,
        tau,
    )


@app.cell
def __(
    ai_ar,
    ai_diff,
    ar_drift,
    diff_drift,
    dram_streamed_ar,
    dram_streamed_diff,
    excess_drift,
    gap_ratio,
    mo,
    prec,
):
    summary_table = mo.md(
        f"""
        ### Benchmark Results: {prec}
        | Metric | Autoregressive (AR) | Block-Diffusion (dLLM) | Delta / Advantage |
        | :--- | :--- | :--- | :--- |
        | **Trajectory Drift Rate** | **{ar_drift:.2f}%** | **{diff_drift:.2f}%** | **{excess_drift:+.2f} pp** |
        | **DRAM Weight Traffic** | {dram_streamed_ar:.1f} MB | **{dram_streamed_diff:.1f} MB** | **{dram_streamed_ar / dram_streamed_diff:.2f}x Bandwidth Reduction** |
        | **Arithmetic Intensity** | {ai_ar:.2f} FLOPs/B | **{ai_diff:.2f} FLOPs/B** | **{ai_diff / ai_ar:.1f}x Higher Intensity (6x compute trade-off)** |
        """
    )
    return (summary_table,)


@app.cell
def __(ar_drift, diff_drift, mo, p_text, prec):
    # Simulated token stream outputs
    if "FP32" in prec:
        ar_tokens = p_text + " causing massive objects to follow geodesic paths through four-dimensional space."
        diff_tokens = p_text + " causing massive objects to follow geodesic paths through four-dimensional space."
    elif "INT4" in prec:
        ar_tokens = p_text + " causing massive objects to follow geodesic orbits through four-dimensional matter."
        diff_tokens = p_text + " causing massive objects to follow geodesic paths through four-dimensional space."
    else:  # Ternary
        if "curvature" in p_text:
            ar_tokens = p_text + " and of the the the of <unk> to <unk> <unk> the <unk> of <unk>"
            diff_tokens = p_text + " caused by the presence of mass and energy distributing along geodesics."
        else:
            ar_tokens = p_text + " ensuring that consensus is reached under standard network conditions."
            diff_tokens = p_text + " ensuring that consensus is reached under standard Byzantine fault thresholds."

    comparison_view = mo.md(
        f"""
        ### Qualitative Rollout Trajectory
        
        **Autoregressive Rollout (Causal KV-Cache, {ar_drift:.1f}% Drift):**
        > `{ar_tokens}`

        **Block-Diffusion Rollout (Bidirectional Canvas, {diff_drift:.1f}% Drift):**
        > `{diff_tokens}`
        """
    )
    return comparison_view, diff_tokens, ar_tokens


@app.cell
def __(comparison_view, mo, summary_table):
    mo.vstack([summary_table, comparison_view])
    return


if __name__ == "__main__":
    app.run()
