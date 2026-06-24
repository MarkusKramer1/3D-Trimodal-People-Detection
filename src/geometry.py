"""Camera calibration and LiDAR↔camera projection for seed painting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation


def load_calibrations(project_root: Path, subset_folder: str) -> dict[str, dict[str, Any]]:
    p = project_root / "data" / subset_folder / "calibrations.json"
    with open(p, encoding="utf-8") as f:
        rows = json.load(f)
    return {c["token"]: c["calibration"] for c in rows}


def se3_matrix_from_calibration(cal: dict[str, Any]) -> np.ndarray:
    Rm = Rotation.from_quat(np.asarray(cal["rotation"], dtype=np.float64)).as_matrix()
    t = np.asarray(cal["translation"], dtype=np.float64).reshape(3)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = Rm
    T[:3, 3] = t
    return T


def camera_to_lidar_from_calibrations(
    lid_cal: dict[str, Any],
    cam_cal: dict[str, Any],
) -> np.ndarray:
    T_cam2world = se3_matrix_from_calibration(cam_cal)
    T_lidar2world = se3_matrix_from_calibration(lid_cal)
    return np.linalg.inv(T_lidar2world) @ T_cam2world


def lidar_to_camera_from_calibrations(
    lid_cal: dict[str, Any],
    cam_cal: dict[str, Any],
) -> np.ndarray:
    return np.linalg.inv(camera_to_lidar_from_calibrations(lid_cal, cam_cal))


def intrinsic_ok(K: np.ndarray) -> bool:
    return float(K[0, 0]) > 1.0 and float(K[1, 1]) > 1.0


def intrinsic_for_actual_image(
    K: np.ndarray,
    nominal_wh: tuple[int, int],
    actual_wh: tuple[int, int],
) -> np.ndarray:
    nw, nh = nominal_wh
    aw, ah = actual_wh
    if nw <= 0 or nh <= 0:
        return K
    sx = aw / float(nw)
    sy = ah / float(nh)
    if abs(sx - 1.0) < 1e-9 and abs(sy - 1.0) < 1e-9:
        return np.array(K, dtype=np.float64, copy=True)
    Ko = np.array(K, dtype=np.float64, copy=True)
    Ko[0, 0] *= sx
    Ko[0, 2] *= sx
    Ko[1, 1] *= sy
    Ko[1, 2] *= sy
    return Ko


def camera_intrinsic_matrix(cal: dict[str, Any]) -> np.ndarray:
    ki = cal.get("camera_intrinisc") or cal.get("camera_intrinsic")
    if ki is None:
        raise KeyError("No camera intrinsic matrix in calibration.")
    return np.asarray(ki, dtype=np.float64).reshape(3, 3)


def distortion_coeffs_from_calibration(cal: dict[str, Any]) -> np.ndarray | None:
    d = cal.get("distortion_coefficients")
    if d is None:
        return None
    return np.asarray(d, dtype=np.float64).reshape(-1)


def undistort_image_rgb_u8(
    rgb_hwc_u8: np.ndarray,
    K_actual: np.ndarray,
    distortion: np.ndarray | None,
) -> tuple[np.ndarray, str]:
    img = np.asarray(rgb_hwc_u8)
    k = np.asarray(K_actual, dtype=np.float64).reshape(3, 3)
    dc = distortion
    if dc is None or np.linalg.norm(np.asarray(dc, dtype=np.float64).ravel()) < 1e-14:
        return img[:, :, :3].copy(), "no distortion"
    if not intrinsic_ok(k):
        raise ValueError("Invalid intrinsic matrix")
    bgr_in = cv2.cvtColor(img[:, :, :3], cv2.COLOR_RGB2BGR)
    dcv = np.asarray(dc, dtype=np.float64).reshape(-1)
    bgr_out = cv2.undistort(bgr_in, k, dcv)
    return cv2.cvtColor(bgr_out, cv2.COLOR_BGR2RGB), "undistorted"


def rectify_pil_rgb_for_calibration(
    pil_rgb: Image.Image,
    cam_cal: dict[str, Any],
    nominal_wh: tuple[int, int],
) -> tuple[Image.Image, np.ndarray]:
    pil_rgb = pil_rgb.convert("RGB")
    arr = np.asarray(pil_rgb, dtype=np.uint8)
    try:
        K_nom = camera_intrinsic_matrix(cam_cal)
    except KeyError:
        return Image.fromarray(arr), np.eye(3, dtype=np.float64)

    w_px, h_px = pil_rgb.size
    K_scaled = intrinsic_for_actual_image(K_nom, nominal_wh, (w_px, h_px))
    if not intrinsic_ok(K_scaled):
        return Image.fromarray(arr), K_scaled

    dist = distortion_coeffs_from_calibration(cam_cal)
    try:
        out_arr, _ = undistort_image_rgb_u8(arr, K_scaled, dist)
    except ValueError:
        return Image.fromarray(arr), K_scaled
    return Image.fromarray(out_arr), K_scaled


def point_to_pixel_z_cam_like_projection(
    points_lidar: np.ndarray,
    lidar_to_camera_4x4: np.ndarray,
    camera_intrinsic: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pts = np.asarray(points_lidar, dtype=np.float32).reshape(-1, 3)
    n_points = int(pts.shape[0])
    if n_points == 0:
        empty_b = np.zeros(0, dtype=bool)
        empty_f = np.zeros((0, 2), dtype=np.float64)
        return empty_f, empty_b, np.zeros(0, dtype=np.float64)

    T = np.asarray(lidar_to_camera_4x4, dtype=np.float32).reshape(4, 4)
    K = np.asarray(camera_intrinsic, dtype=np.float32).reshape(3, 3)
    R = T[:3, :3]
    t = T[:3, 3]
    points_cam = pts @ R.T + t
    z_cam = points_cam[:, 2].astype(np.float64, copy=False)
    valid_mask = points_cam[:, 2] > 0.0
    pixels = np.zeros((n_points, 2), dtype=np.float64)
    if np.any(valid_mask):
        pc = points_cam[valid_mask]
        inv_z = 1.0 / pc[:, 2]
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        pixels[valid_mask, 0] = (pc[:, 0] * inv_z * fx + cx).astype(np.float64, copy=False)
        pixels[valid_mask, 1] = (pc[:, 1] * inv_z * fy + cy).astype(np.float64, copy=False)
    return pixels, valid_mask, z_cam
