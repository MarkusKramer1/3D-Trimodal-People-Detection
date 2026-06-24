"""Interactive demo visualization: mask overlays + Open3D point cloud in one window."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

try:
    import open3d as o3d
    import open3d.visualization.gui as gui
    import open3d.visualization.rendering as rendering
except ImportError:  # pragma: no cover
    o3d = None  # type: ignore[assignment]
    gui = None  # type: ignore[assignment]
    rendering = None  # type: ignore[assignment]

MaskList = list[tuple[int, np.ndarray, float]]

_MASK_COLORS = np.array(
    [
        [0.12, 0.85, 0.45],
        [1.00, 0.55, 0.00],
        [0.35, 0.55, 1.00],
        [0.95, 0.25, 0.55],
        [0.75, 0.35, 0.95],
        [0.95, 0.85, 0.15],
    ],
    dtype=np.float64,
)


def subsample_indices(n: int, max_points: int, seed: int = 0) -> np.ndarray:
    if n <= max_points:
        return np.arange(n, dtype=np.intp)
    rng = np.random.default_rng(seed)
    return rng.choice(n, max_points, replace=False)


def overlay_segmentation_masks(
    image_rgb: np.ndarray,
    masks: MaskList,
    *,
    alpha: float = 0.45,
) -> np.ndarray:
    """Blend person instance masks onto an RGB image."""
    base = np.asarray(image_rgb, dtype=np.float32)
    if base.max() > 1.5:
        base = base / 255.0
    out = base.copy()
    for i, (_, mask, _score) in enumerate(masks):
        color = _MASK_COLORS[i % len(_MASK_COLORS)]
        m = np.asarray(mask, dtype=bool)
        if not m.any():
            continue
        out[m] = out[m] * (1.0 - alpha) + color * alpha
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


def _lidar_box_corners(box: Sequence[float]) -> np.ndarray:
    """LiDAR box bottom-center ``[x,y,z,dx,dy,dz,yaw]`` → ``(8,3)`` corners."""
    x, y, z, dx, dy, dz, yaw = box[:7]
    c, s = float(np.cos(yaw)), float(np.sin(yaw))
    hx, hy, dz_val = dx * 0.5, dy * 0.5, dz
    local = np.array(
        [
            [hx, hy, 0],
            [hx, -hy, 0],
            [-hx, -hy, 0],
            [-hx, hy, 0],
            [hx, hy, dz_val],
            [hx, -hy, dz_val],
            [-hx, -hy, dz_val],
            [-hx, hy, dz_val],
        ],
        dtype=np.float64,
    )
    rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)
    return local @ rot.T + np.array([x, y, z], dtype=np.float64)


_BOX_EDGES = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
)


def detection_box_line_sets(
    detections: Any,
    *,
    score_thr: float,
    color: tuple[float, float, float] = (0.92, 0.20, 0.20),
) -> list[Any]:
    """Open3D line sets for 3D detection boxes above ``score_thr``."""
    if o3d is None:
        raise ImportError("open3d is required for 3D visualization")

    geoms: list[Any] = []
    for score, box in zip(detections.scores_3d, detections.bboxes_3d):
        if score < score_thr:
            continue
        corners = _lidar_box_corners(box)
        lines = o3d.geometry.LineSet()
        lines.points = o3d.utility.Vector3dVector(corners)
        lines.lines = o3d.utility.Vector2iVector(list(_BOX_EDGES))
        lines.paint_uniform_color(color)
        geoms.append(lines)
    return geoms


def detection_focus_bounds(
    detections: Any,
    *,
    score_thr: float,
    padding_m: float = 1.5,
) -> Any | None:
    """Axis-aligned bounds around all detection boxes (for camera framing)."""
    if o3d is None:
        return None

    corners: list[np.ndarray] = []
    for score, box in zip(detections.scores_3d, detections.bboxes_3d):
        if score < score_thr:
            continue
        corners.append(_lidar_box_corners(box))
    if not corners:
        return None

    all_corners = np.vstack(corners)
    pad = float(padding_m)
    return o3d.geometry.AxisAlignedBoundingBox(
        all_corners.min(axis=0) - pad,
        all_corners.max(axis=0) + pad,
    )


def _view_from_sensor_origin(
    scene_widget: Any,
    *,
    focus_bb: Any | None,
    sensor_origin: np.ndarray | Sequence[float] = (0.0, 0.0, 0.0),
    vfov_deg: float = 45.0,
) -> None:
    """Camera near the LiDAR origin: slightly raised and oblique into the scene (−x)."""
    origin = np.asarray(sensor_origin, dtype=np.float64).reshape(3)
    # Small offset from the sensor: higher (+z) and lateral (+y) for a clearer 3/4 view.
    eye = origin + np.array([0.0, 1.4, 2.0], dtype=np.float64)
    if focus_bb is not None and float(np.linalg.norm(focus_bb.get_extent())) > 0:
        target = np.asarray(focus_bb.get_center(), dtype=np.float64)
        scene_widget.setup_camera(vfov_deg, focus_bb, target)
    else:
        target = np.array([-12.0, 0.0, 0.0], dtype=np.float64)
        fallback = o3d.geometry.AxisAlignedBoundingBox(
            target - np.array([8.0, 8.0, 4.0]),
            target + np.array([8.0, 8.0, 4.0]),
        )
        scene_widget.setup_camera(vfov_deg, fallback, target)

    scene_widget.scene.camera.look_at(target, eye, [0.0, 0.0, 1.0])


def build_point_cloud_geometry(
    xyz_real: np.ndarray,
    xyz_virtual: np.ndarray,
    *,
    max_points: int,
) -> Any:
    """Colored Open3D point cloud (gray LiDAR + cyan virtual points)."""
    if o3d is None:
        raise ImportError("open3d is required for 3D visualization")

    chunks_xyz: list[np.ndarray] = []
    chunks_rgb: list[np.ndarray] = []
    if xyz_real.shape[0] > 0:
        idx = subsample_indices(xyz_real.shape[0], max_points)
        chunks_xyz.append(xyz_real[idx])
        chunks_rgb.append(np.tile([0.55, 0.55, 0.55], (idx.shape[0], 1)))
    if xyz_virtual.shape[0] > 0:
        idx = subsample_indices(xyz_virtual.shape[0], max_points, seed=1)
        chunks_xyz.append(xyz_virtual[idx])
        chunks_rgb.append(np.tile([0.0, 0.78, 0.85], (idx.shape[0], 1)))

    xyz = np.vstack(chunks_xyz) if chunks_xyz else np.zeros((0, 3), dtype=np.float64)
    rgb = np.vstack(chunks_rgb) if chunks_rgb else np.zeros((0, 3), dtype=np.float64)

    pcd = o3d.geometry.PointCloud()
    if xyz.shape[0] > 0:
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.colors = o3d.utility.Vector3dVector(rgb)
    return pcd


def _as_o3d_image(image_rgb: np.ndarray) -> Any:
    arr = np.ascontiguousarray(image_rgb, dtype=np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    return o3d.geometry.Image(arr)


def show_demo_window(
    *,
    images: dict[str, tuple[np.ndarray, int, int] | None],
    inline_masks: dict[str, MaskList],
    xyz_real: np.ndarray,
    xyz_virtual: np.ndarray,
    detections: Any,
    score_thr: float,
    max_points: int,
    window_title: str,
) -> None:
    """Single Open3D window: camera images with masks (left) + interactive 3D (right)."""
    if o3d is None or gui is None or rendering is None:
        raise ImportError("open3d is required; pip install open3d")

    app = gui.Application.instance
    app.initialize()

    window = app.create_window(window_title, 1600, 900)
    em = window.theme.font_size

    scene = gui.SceneWidget()
    scene.scene = rendering.Open3DScene(window.renderer)
    scene.scene.set_background([0.08, 0.09, 0.11, 1.0])
    scene.enable_scene_caching(True)

    rgb_widget = gui.ImageWidget()
    thermal_widget = gui.ImageWidget()

    for widget, mod in ((rgb_widget, "rgb"), (thermal_widget, "thermal")):
        loaded = images.get(mod)
        if loaded is None:
            placeholder = np.zeros((64, 64, 3), dtype=np.uint8)
            widget.update_image(_as_o3d_image(placeholder))
            continue
        image_rgb, _, _ = loaded
        shown = overlay_segmentation_masks(image_rgb, inline_masks.get(mod, []))
        widget.update_image(_as_o3d_image(shown))

    images_panel = gui.Vert(0.5 * em, gui.Margins(int(0.5 * em), int(0.5 * em), int(0.5 * em), int(0.5 * em)))
    images_panel.add_child(gui.Label("RGB + YOLO masks"))
    images_panel.add_child(rgb_widget)
    images_panel.add_child(gui.Label("Thermal + YOLO masks"))
    images_panel.add_child(thermal_widget)

    # Direct window children only — on_layout must set frames on these, not on a wrapper.
    window.add_child(scene)
    window.add_child(images_panel)

    pcd = build_point_cloud_geometry(xyz_real, xyz_virtual, max_points=max_points)
    pcd_mat = rendering.MaterialRecord()
    pcd_mat.shader = "defaultUnlit"
    pcd_mat.point_size = max(3, int(3 * window.scaling))
    scene.scene.add_geometry("points", pcd, pcd_mat)

    line_mat = rendering.MaterialRecord()
    line_mat.shader = "unlitLine"
    line_mat.line_width = 2.0
    for i, line_set in enumerate(detection_box_line_sets(detections, score_thr=score_thr)):
        scene.scene.add_geometry(f"det_{i}", line_set, line_mat)

    focus_bb = detection_focus_bounds(detections, score_thr=score_thr, padding_m=1.5)
    camera_ready = {"done": False}

    def on_layout(layout_context: Any) -> None:
        content = window.content_rect
        panel_w = min(420, max(int(14 * layout_context.theme.font_size), int(content.width * 0.30)))
        images_panel.frame = gui.Rect(content.x, content.y, panel_w, content.height)
        scene.frame = gui.Rect(content.x + panel_w, content.y, content.width - panel_w, content.height)

        if not camera_ready["done"] and scene.frame.width > 0:
            _view_from_sensor_origin(scene, focus_bb=focus_bb)
            camera_ready["done"] = True
            scene.force_redraw()

    window.set_on_layout(on_layout)

    print(
        "Demo window open — left: RGB/thermal with segmentation masks, "
        "right: interactive point cloud (drag to rotate, scroll to zoom). "
        "Close the window to exit."
    )
    app.run()
