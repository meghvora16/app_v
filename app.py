"""
Streamlit front-end for the Ecorr-anchored automatic active-region Tafel fit.

Run with:   streamlit run app.py

Upload one or more Gamry .DTA files (or txt/csv/xlsx exports). Each file is
treated like one "point" folder of the original batch script.
"""

import io
import shutil
import tempfile
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import tafel_core as core

st.set_page_config(page_title="Tafel Fit", layout="wide")
st.title("Ecorr-anchored automatic Tafel fit")
st.caption(
    "Upload Gamry .DTA files (LSV / potentiodynamic). The fitting logic is the "
    "original batch script, unchanged; each uploaded file is one 'point'."
)

# =============================================================================
# SIDEBAR — the USER SETTINGS block of the original script
# =============================================================================
with st.sidebar:
    st.header("Settings")

    st.subheader("Electrode area")
    use_header_area = st.checkbox(
        "Use AREA from each .DTA header",
        value=False,
        help="Off = use the value below for every file (original behaviour).",
    )
    area = st.number_input(
        "Electrode area (cm²)", value=1.0, min_value=1e-9, format="%.4f",
        help="Use 1.0 if current is already current density.",
    )

    st.subheader("Data")
    every_nth = st.number_input(
        "Use every Nth data point", value=1, min_value=1, step=1,
        help=(
            "1 = all points (original behaviour). High-resolution Gamry scans "
            "(e.g. 0.1 mV steps) may need thinning so that a window of at most "
            "'Max points per window' can span the minimum potential span."
        ),
    )

    st.subheader("Active-region constraints")
    min_points = st.number_input("Min points per branch", value=6, min_value=2, step=1)
    min_span_mv = st.number_input("Min potential span (mV)", value=20.0, min_value=0.0)
    min_log_span = st.number_input("Min log-current span (decades)", value=0.30, min_value=0.0, format="%.2f")
    min_r2 = st.number_input("Min R² for active region", value=0.940, min_value=0.0, max_value=1.0, format="%.3f")
    excl_mv = st.number_input("Ecorr exclusion window (± mV)", value=10.0, min_value=0.0)
    unlimited = st.checkbox("No limit on points per window", value=False)
    max_points = st.number_input(
        "Max points per window", value=80, min_value=2, step=1, disabled=unlimited
    )
    min_abs_j = st.number_input("Min |j| used for fitting (A/cm²)", value=1e-10, format="%.1e")

    st.subheader("Ecorr anchoring")
    max_shift_mv = st.number_input("Max Ecorr shift (± mV)", value=15.0, min_value=0.0)
    match_weight = st.number_input("Ecorr match weight", value=1000.0, min_value=0.0)
    allow_fallback = st.checkbox("Allow Ecorr fallback", value=True)

    st.subheader("Plot")
    show_full = st.checkbox("Show full curve in background", value=True)
    png_dpi = st.number_input("PNG DPI", value=300, min_value=50, step=50)

    fast_mode = st.checkbox(
        "Vectorised computation (recommended)",
        value=True,
        help=(
            "Same maths and same selection as the original loops, computed with "
            "NumPy arrays. Turn off to run the original pure-Python loops "
            "(much slower on large files)."
        ),
    )


def apply_settings():
    core.MIN_POINTS_PER_BRANCH = int(min_points)
    core.MIN_POTENTIAL_SPAN_V = min_span_mv / 1000.0
    core.MIN_LOG_CURRENT_SPAN_DECADES = float(min_log_span)
    core.MIN_R2_FOR_ACTIVE_REGION = float(min_r2)
    core.ECORR_EXCLUSION_WINDOW_V = excl_mv / 1000.0
    core.MAX_POINTS_PER_WINDOW = None if unlimited else int(max_points)
    core.MIN_ABS_CURRENT_DENSITY_A_CM2 = float(min_abs_j)
    core.MAX_ECORR_SHIFT_V = max_shift_mv / 1000.0
    core.ECORR_MATCH_WEIGHT = float(match_weight)
    core.ALLOW_ECORR_FALLBACK = bool(allow_fallback)
    core.SHOW_FULL_CURVE_IN_BACKGROUND = bool(show_full)
    core.PNG_DPI = int(png_dpi)
    core.ELECTRODE_AREA_CM2 = float(area)
    core.set_fast_mode(bool(fast_mode))


# =============================================================================
# FILE READING
# =============================================================================

def read_uploaded(name: str, data: bytes, workdir: Path):
    """Return (dataframe, header_dict). .DTA → Gamry parser, else original loader."""
    if name.lower().endswith(".dta"):
        text = data.decode("latin-1")
        return core.load_gamry_dta(text)

    tmp = workdir / name
    tmp.write_bytes(data)
    return core.load_lsv_file(tmp), {}


def process_file(name: str, data: bytes, out_root: Path) -> dict:
    """Same steps as process_one_folder(), with the uploaded file as the source."""
    stem = Path(name).stem
    folder = out_root / stem
    folder.mkdir(parents=True, exist_ok=True)
    source_file = Path(name)

    try:
        dataframe, header = read_uploaded(name, data, out_root)

        file_area = core.ELECTRODE_AREA_CM2
        if use_header_area and "AREA" in header:
            file_area = float(header["AREA"])
        core.ELECTRODE_AREA_CM2 = file_area

        potential_column, current_column, current_factor = core.auto_detect_columns(dataframe)

        potential = pd.to_numeric(dataframe[potential_column], errors="coerce").to_numpy()
        current_a = (
            pd.to_numeric(dataframe[current_column], errors="coerce").to_numpy()
            * current_factor
        )

        valid = np.isfinite(potential) & np.isfinite(current_a)
        potential = potential[valid][:: int(every_nth)]
        current_a = current_a[valid][:: int(every_nth)]

        if len(potential) < 10:
            raise ValueError("Fewer than 10 valid potential/current points found.")

        current_density = current_a / core.ELECTRODE_AREA_CM2

        fit = core.fit_active_tafel_region(potential, current_density)

        result_row = core.create_result_row(
            folder, source_file, potential_column, current_column, current_factor, fit
        )

        figure = core.save_active_tafel_plot(folder, potential, current_density, fit)
        figure.savefig(folder / "tafel_active_fit.png", dpi=core.PNG_DPI, bbox_inches="tight")
        plt.close(figure)

        core.save_individual_results(folder, potential, current_density, result_row, fit)

        return {"row": result_row, "folder": folder, "ok": True}

    except Exception as error:
        return {
            "row": {
                "Folder": stem,
                "Source file": name,
                "Status": f"Error: {type(error).__name__}: {error}",
            },
            "folder": folder,
            "ok": False,
        }
    finally:
        core.ELECTRODE_AREA_CM2 = float(area)


def precheck(name: str, data: bytes, workdir: Path):
    """Warn if the window settings make a valid window impossible."""
    try:
        dataframe, _ = read_uploaded(name, data, workdir)
        pcol, _, _ = core.auto_detect_columns(dataframe)
        e = pd.to_numeric(dataframe[pcol], errors="coerce").dropna().to_numpy()[:: int(every_nth)]
        step = float(np.median(np.abs(np.diff(e))))
        if unlimited or step <= 0:
            return None
        reachable = (int(max_points) - 1) * step
        if reachable < min_span_mv / 1000.0:
            need_n = int(np.ceil((min_span_mv / 1000.0) / ((int(max_points) - 1) * step / int(every_nth))))
            return (
                f"**{name}**: potential step ≈ {step*1000:.3g} mV, so a window of "
                f"≤{int(max_points)} points spans at most {reachable*1000:.1f} mV — "
                f"below the {min_span_mv:.0f} mV minimum. No window can pass. "
                f"Increase *Use every Nth data point* (≥ {need_n}) or *Max points per window*."
            )
    except Exception:
        return None
    return None


# =============================================================================
# MAIN
# =============================================================================

uploads = st.file_uploader(
    "Upload data files",
    type=["dta", "txt", "csv", "xlsx", "xls"],
    accept_multiple_files=True,
)

if uploads:
    with tempfile.TemporaryDirectory() as tmp:
        warnings_ = [precheck(u.name, u.getvalue(), Path(tmp)) for u in uploads]
    for w in filter(None, warnings_):
        st.warning(w)

run = st.button("Run Tafel fit", type="primary", disabled=not uploads)

if run and uploads:
    apply_settings()

    out_root = Path(tempfile.mkdtemp(prefix="tafel_"))
    core.ASCII_FOLDER = out_root

    results = []
    progress = st.progress(0.0, text="Fitting…")
    for i, upload in enumerate(uploads, start=1):
        progress.progress((i - 1) / len(uploads), text=f"Fitting {upload.name} ({i}/{len(uploads)})")
        results.append(process_file(upload.name, upload.getvalue(), out_root))
    progress.progress(1.0, text="Done")

    summary_rows = [r["row"] for r in results]
    core.save_summary(summary_rows)

    # Collect everything into memory (temp folder is removed afterwards)
    files = {}
    for path in out_root.rglob("*"):
        if path.is_file() and path.name.startswith(("tafel_active", "Tafel Summary")):
            files[str(path.relative_to(out_root))] = path.read_bytes()

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel, content in files.items():
            zf.writestr(rel, content)

    st.session_state["results"] = {
        "summary": pd.DataFrame(summary_rows),
        "items": [
            {"name": r["folder"].name, "ok": r["ok"], "row": r["row"]} for r in results
        ],
        "files": files,
        "zip": zip_buffer.getvalue(),
    }
    shutil.rmtree(out_root, ignore_errors=True)

# =============================================================================
# RESULTS
# =============================================================================

res = st.session_state.get("results")
if res:
    ok_count = sum(item["ok"] for item in res["items"])
    st.subheader(f"Summary — {ok_count}/{len(res['items'])} successful")

    st.dataframe(res["summary"], width="stretch")

    c1, c2, c3 = st.columns(3)
    c1.download_button(
        "Download everything (.zip)", res["zip"], "tafel_results.zip", "application/zip"
    )
    if "Tafel Summary.xlsx" in res["files"]:
        c2.download_button(
            "Tafel Summary.xlsx", res["files"]["Tafel Summary.xlsx"], "Tafel Summary.xlsx"
        )
    if "Tafel Summary.csv" in res["files"]:
        c3.download_button(
            "Tafel Summary.csv", res["files"]["Tafel Summary.csv"], "Tafel Summary.csv", "text/csv"
        )

    tabs = st.tabs([item["name"] for item in res["items"]])
    for tab, item in zip(tabs, res["items"]):
        with tab:
            row = item["row"]
            if not item["ok"]:
                st.error(row["Status"])
                continue

            if row["Ecorr constraint met"]:
                st.success(row["Status"])
            else:
                st.warning(row["Status"] + " — fitted Ecorr is outside the allowed shift.")

            m = st.columns(5)
            m[0].metric("icorr (A/cm²)", f"{row['icorr (A/cm²)']:.3e}")
            m[1].metric("Ecorr fitted (V)", f"{row['Ecorr fitted Tafel intersection (V)']:.5f}")
            m[2].metric("ΔEcorr (mV)", f"{row['Ecorr difference fitted - observed (mV)']:+.1f}")
            m[3].metric("βa (mV/dec)", f"{row['Anodic Tafel slope βa (mV/dec)']:.1f}",
                        help=f"R² = {row['Anodic R²']:.4f}")
            m[4].metric("βc (mV/dec)", f"{row['Cathodic Tafel slope βc (mV/dec)']:.1f}",
                        help=f"R² = {row['Cathodic R²']:.4f}")

            name = item["name"]
            png = res["files"].get(f"{name}/tafel_active_fit.png")
            if png:
                st.image(png, width="stretch")

            with st.expander("All fit parameters"):
                st.dataframe(
                    pd.DataFrame({"Parameter": list(row.keys()),
                                  "Value": [str(v) for v in row.values()]}),
                    width="stretch", hide_index=True,
                )

            d = st.columns(4)
            for col, fname, mime in [
                (d[0], "tafel_active_fit.png", "image/png"),
                (d[1], "tafel_active_results.xlsx",
                 "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
                (d[2], "tafel_active_results.csv", "text/csv"),
                (d[3], "tafel_active_data.csv", "text/csv"),
            ]:
                key = f"{name}/{fname}"
                if key in res["files"]:
                    col.download_button(fname, res["files"][key], f"{name}_{fname}", mime,
                                        key=f"dl_{key}")
