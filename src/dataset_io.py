"""Load scene metadata and LiDAR point clouds for the reference pipeline."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import numpy as np


def load_links(project_root: Path, subset_folder: str) -> list[dict[str, Any]]:
    p = project_root / "data" / subset_folder / "links.json"
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def load_samples(project_root: Path, subset_folder: str) -> list[dict[str, Any]]:
    p = project_root / "data" / subset_folder / "samples.json"
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def sample_full_path(project_root: Path, subset_folder: str, filename: str) -> Path:
    rel = filename.lstrip("/")
    return (project_root / "data" / subset_folder / "samples" / rel).resolve()


def build_token_lookup(samples: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {s["token"]: s for s in samples}


def link_token_field(modality: str) -> str:
    if modality == "rgb":
        return "rgb_token"
    if modality == "thermal":
        return "thermal_token"
    raise ValueError(f"unsupported modality: {modality}")


def _parse_pcd_header(path: Path) -> tuple[dict[str, object], int]:
    header: dict[str, object] = {"fields": [], "sizes": [], "types": [], "points": 0, "data": None}
    data_offset = 0
    with open(path, "rb") as f:
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"Invalid PCD header: {path}")
            text = line.decode("ascii", errors="replace").strip()
            if text.startswith("FIELDS"):
                header["fields"] = text.split()[1:]
            elif text.startswith("SIZE"):
                header["sizes"] = [int(x) for x in text.split()[1:]]
            elif text.startswith("TYPE"):
                header["types"] = text.split()[1:]
            elif text.startswith("POINTS"):
                header["points"] = int(text.split()[1])
            elif text.startswith("DATA"):
                header["data"] = text.split()[1].lower()
                data_offset = f.tell()
                break
    return header, data_offset


def read_pcd_points_ascii(path: Path) -> np.ndarray:
    text = path.read_text(encoding="utf-8", errors="replace")
    if "DATA ascii" not in text:
        raise ValueError(f"Unsupported PCD (need ASCII): {path}")
    _, rest = text.split("DATA ascii", 1)
    rest = rest.strip()
    if not rest:
        return np.zeros((0, 3))
    try:
        import pandas as pd

        df = pd.read_csv(
            io.StringIO(rest),
            sep=" ",
            header=None,
            engine="c",
            usecols=(0, 1, 2),
            dtype=np.float64,
            on_bad_lines="skip",
        )
        raw = df.to_numpy(dtype=np.float64, copy=False)
    except Exception:
        raw = np.loadtxt(io.StringIO(rest), dtype=np.float64, usecols=(0, 1, 2))
    if raw.size == 0:
        return np.zeros((0, 3))
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    return raw[:, :3]


def read_pcd_xyz_intensity(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read PCD xyz and optional intensity; intensity is zeros when the field is absent."""
    header, data_offset = _parse_pcd_header(path)
    data_kind = str(header["data"] or "")
    if data_kind == "ascii":
        xyz = read_pcd_points_ascii(path)
        intensity = np.zeros(xyz.shape[0], dtype=np.float64)
        text = path.read_text(encoding="utf-8", errors="replace")
        if "DATA ascii" not in text:
            return xyz, intensity
        _, rest = text.split("DATA ascii", 1)
        rest = rest.strip()
        if not rest:
            return xyz, intensity
        fields = list(header["fields"])
        if "intensity" not in fields:
            return xyz, intensity
        idx = fields.index("intensity")
        try:
            import pandas as pd

            df = pd.read_csv(
                io.StringIO(rest),
                sep=" ",
                header=None,
                engine="c",
                usecols=(idx,),
                dtype=np.float64,
                on_bad_lines="skip",
            )
            intensity = df.to_numpy(dtype=np.float64, copy=False).reshape(-1)
        except Exception:
            raw = np.loadtxt(io.StringIO(rest), dtype=np.float64)
            if raw.ndim == 1:
                raw = raw.reshape(1, -1)
            intensity = raw[:, idx] if raw.shape[1] > idx else np.zeros(0, dtype=np.float64)
        n = min(xyz.shape[0], intensity.shape[0])
        return xyz[:n], np.asarray(intensity[:n], dtype=np.float64)

    if data_kind != "binary":
        raise ValueError(f"Unsupported PCD DATA {data_kind!r}: {path}")

    fields = list(header["fields"])
    sizes = list(header["sizes"])
    types = list(header["types"])
    n_points = int(header["points"])
    if n_points <= 0 or not fields:
        return np.zeros((0, 3), dtype=np.float64), np.zeros(0, dtype=np.float64)
    if fields[:3] != ["x", "y", "z"]:
        raise ValueError(f"PCD must start with x y z fields: {path}")

    point_step = sum(int(s) for s in sizes)
    raw = path.read_bytes()[data_offset : data_offset + n_points * point_step]
    if len(raw) < n_points * point_step:
        raise ValueError(f"Truncated binary PCD payload: {path}")

    dtype_fields: list[tuple[str, str]] = []
    for field, size, typ in zip(fields, sizes, types):
        if typ == "F" and size == 4:
            dtype_fields.append((field, "<f4"))
        elif typ == "F" and size == 8:
            dtype_fields.append((field, "<f8"))
        elif typ == "U" and size == 1:
            dtype_fields.append((field, "u1"))
        elif typ == "U" and size == 2:
            dtype_fields.append((field, "<u2"))
        elif typ == "U" and size == 4:
            dtype_fields.append((field, "<u4"))
        else:
            raise ValueError(f"Unsupported PCD field {field} TYPE {typ} SIZE {size}: {path}")
    structured = np.frombuffer(raw, dtype=np.dtype(dtype_fields), count=n_points)
    xyz = np.column_stack(
        (
            np.asarray(structured["x"], dtype=np.float64),
            np.asarray(structured["y"], dtype=np.float64),
            np.asarray(structured["z"], dtype=np.float64),
        )
    )
    if "intensity" in structured.dtype.names:
        intensity = np.asarray(structured["intensity"], dtype=np.float64)
    else:
        intensity = np.zeros(n_points, dtype=np.float64)
    return xyz, intensity
