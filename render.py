"""Inspection images: raw left view next to the predicted hand overlay."""
import cv2
import numpy as np

EDGES = [(0, 1+4*f) for f in range(5)] + [(1+4*f+j, 2+4*f+j) for f in range(5) for j in range(3)]
COLORS = [(255, 170, 40), (50, 220, 255), (80, 230, 80), (220, 100, 255), (255, 210, 100)]


def render_comparison(image, hands, title, *, panel_width=800, mesh=False, faces=None):
    scale = panel_width / image.shape[1]
    size = (panel_width, round(image.shape[0] * scale))
    raw = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    overlay = raw.copy()
    for hand in hands:
        right = bool(hand["right"])
        color = (80, 220, 90) if right else (60, 170, 255)
        if mesh and faces is not None:
            vertices = np.asarray(hand["vertices_2d"]) * scale
            layer = overlay.copy()
            for face in faces:
                pts = vertices[face]
                if np.isfinite(pts).all():
                    cv2.polylines(layer, [np.rint(pts).astype(np.int32)], True, color, 1, cv2.LINE_AA)
            overlay = cv2.addWeighted(layer, .35, overlay, .65, 0)
        joints = np.asarray(hand["keypoints_2d"]) * scale
        for a, b in EDGES:
            if np.isfinite(joints[[a, b]]).all():
                f = (b-1) // 4
                cv2.line(overlay, tuple(np.rint(joints[a]).astype(int)), tuple(np.rint(joints[b]).astype(int)), COLORS[f], 2, cv2.LINE_AA)
        for point in joints:
            if np.isfinite(point).all():
                p = tuple(np.rint(point).astype(int))
                cv2.circle(overlay, p, 3, (25, 25, 25), -1, cv2.LINE_AA)
                cv2.circle(overlay, p, 2, color, -1, cv2.LINE_AA)
        x0, y0, x1, y1 = np.rint(np.asarray(hand["bbox"]) * scale).astype(int)
        cv2.rectangle(overlay, (x0, y0), (x1, y1), color, 1, cv2.LINE_AA)
        source = hand.get("track_source", "detector").replace("visual_tracker", "tracked")
        label = f"{'R' if right else 'L'}* {source} {hand['detector_score']:.2f}"
        age = hand.get("seconds_since_detection", 0.)
        if age > .001:
            label += f" +{age:.2f}s"
        cv2.putText(overlay, label, (max(4, x0), max(20, y0-8)), cv2.FONT_HERSHEY_SIMPLEX, .5, color, 1, cv2.LINE_AA)
    if not hands:
        cv2.putText(overlay, "No hand detected", (22, 40), cv2.FONT_HERSHEY_SIMPLEX, .8, (70, 170, 255), 2, cv2.LINE_AA)
    header = np.full((70, 2*panel_width, 3), (28, 28, 28), np.uint8)
    cv2.putText(header, title[:120], (18, 27), cv2.FONT_HERSHEY_SIMPLEX, .65, (245, 245, 245), 1, cv2.LINE_AA)
    cv2.putText(header, "LEFT EGO CAMERA", (18, 56), cv2.FONT_HERSHEY_SIMPLEX, .48, (170, 170, 170), 1, cv2.LINE_AA)
    tracked = any("track_id" in hand for hand in hands)
    subtitle = "AnyHand-WiLoR | persistent tracks | L/R* unverified" if tracked else "AnyHand-WiLoR | L/R* = ego-position heuristic"
    cv2.putText(header, subtitle, (panel_width+18, 56), cv2.FONT_HERSHEY_SIMPLEX, .48, (210, 210, 210), 1, cv2.LINE_AA)
    return np.vstack([header, np.hstack([raw, overlay])])


def contact_sheet(images, width=1200):
    resized = [cv2.resize(img, (width, round(img.shape[0]*width/img.shape[1])), interpolation=cv2.INTER_AREA) for img in images]
    return np.vstack(resized)
