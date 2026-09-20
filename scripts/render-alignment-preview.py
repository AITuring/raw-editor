#!/usr/bin/env python3
"""Render a quick, source-only preview from a panorama alignment cache."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


class DisjointSet:
    def __init__(self, values):
        self.parent = {value: value for value in values}

    def find(self, value):
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left, right):
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def transformed_point(matrix, point):
    mapped = matrix @ point
    return mapped[:2] / mapped[2]


def source_preview_path(source, image_dir):
    stem = Path(source["filename"]).stem
    return image_dir / f"{source['id']:04d}-{stem}.jpg"


def capture_groups(cache, sources, homographies):
    ids = [source["id"] for source in cache["sources"]]
    groups = DisjointSet(ids)
    coarse_boundaries = set()
    for match in cache["matches"]:
        if match.get("local_bracket"):
            groups.union(match["source_id"], match["target_id"])
        if match.get("coarse_bridge"):
            coarse_boundaries.add(
                tuple(sorted((match["source_id"], match["target_id"])))
            )

    ordered = sorted(ids, key=lambda source_id: sources[source_id]["filename"])
    for left, right in zip(ordered, ordered[1:]):
        if (left, right) in coarse_boundaries:
            continue
        left_source = sources[left]
        right_source = sources[right]
        left_center = transformed_point(
            homographies[left],
            np.array(
                [left_source["width"] * 0.5, left_source["height"] * 0.5, 1.0]
            ),
        )
        right_center = transformed_point(
            homographies[right],
            np.array(
                [right_source["width"] * 0.5, right_source["height"] * 0.5, 1.0]
            ),
        )
        scale = max(
            left_source["width"],
            left_source["height"],
            right_source["width"],
            right_source["height"],
            1,
        )
        if np.linalg.norm(left_center - right_center) / scale <= 0.075:
            groups.union(left, right)

    components = {}
    for source_id in ids:
        components.setdefault(groups.find(source_id), []).append(source_id)
    split_components = []
    for component in components.values():
        component.sort(key=lambda source_id: sources[source_id]["filename"])
        current = []
        anchor_center = None
        for source_id in component:
            source = sources[source_id]
            center = transformed_point(
                homographies[source_id],
                np.array([source["width"] * 0.5, source["height"] * 0.5, 1.0]),
            )
            scale = max(source["width"], source["height"], 1)
            if (
                current
                and (
                    len(current) >= 48
                    or np.linalg.norm(anchor_center - center) / scale > 0.075
                )
            ):
                split_components.append(current)
                current = []
                anchor_center = None
            if not current:
                anchor_center = center
            current.append(source_id)
        if current:
            split_components.append(current)
    return sorted(split_components, key=lambda group: min(group))


def sharpest_source(group, sources, image_dir):
    best = None
    for source_id in group:
        path = source_preview_path(sources[source_id], image_dir)
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        height, width = image.shape
        roi = image[height // 8 : height * 7 // 8, width // 8 : width * 7 // 8]
        score = cv2.Laplacian(roi, cv2.CV_32F).var()
        if best is None or score > best[0]:
            best = (score, source_id)
    if best is None:
        raise RuntimeError(f"No preview image found for capture group {group}")
    return best[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cache", type=Path)
    parser.add_argument("image_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--width", type=int, default=3000)
    args = parser.parse_args()

    cache = json.loads(args.cache.read_text())
    sources = {source["id"]: source for source in cache["sources"]}
    homographies = {
        int(source_id): np.asarray(values, dtype=np.float64).reshape(3, 3)
        for source_id, values in cache["global_homographies"].items()
    }
    selected = [
        sharpest_source(group, sources, args.image_dir)
        for group in capture_groups(cache, sources, homographies)
    ]

    corners = []
    for source_id in selected:
        source = sources[source_id]
        matrix = homographies[source_id]
        for x, y in (
            (0.0, 0.0),
            (source["width"], 0.0),
            (source["width"], source["height"]),
            (0.0, source["height"]),
        ):
            corners.append(transformed_point(matrix, np.array([x, y, 1.0])))
    corners = np.asarray(corners)
    minimum = np.floor(corners.min(axis=0))
    maximum = np.ceil(corners.max(axis=0))
    full_size = maximum - minimum
    scale = min(1.0, args.width / full_size[0])
    output_size = tuple(np.maximum(1, np.ceil(full_size * scale)).astype(int))
    canvas = np.zeros((output_size[1], output_size[0], 3), dtype=np.float32)
    weights = np.zeros((output_size[1], output_size[0]), dtype=np.float32)
    offset = np.array(
        [[1.0, 0.0, -minimum[0]], [0.0, 1.0, -minimum[1]], [0.0, 0.0, 1.0]]
    )
    output_scale = np.diag([scale, scale, 1.0])

    for source_id in selected:
        source = sources[source_id]
        image = cv2.imread(
            str(source_preview_path(source, args.image_dir)), cv2.IMREAD_COLOR
        )
        if image is None:
            continue
        source_scale = np.diag(
            [source["width"] / image.shape[1], source["height"] / image.shape[0], 1.0]
        )
        transform = output_scale @ offset @ homographies[source_id] @ source_scale
        mask = np.full(image.shape[:2], 255, dtype=np.uint8)
        warped = cv2.warpPerspective(
            image, transform, output_size, flags=cv2.INTER_LINEAR
        )
        warped_mask = cv2.warpPerspective(
            mask, transform, output_size, flags=cv2.INTER_NEAREST
        )
        distance = cv2.distanceTransform(warped_mask, cv2.DIST_L2, 3)
        layer_weight = np.minimum(distance + 1.0, 48.0)
        layer_weight[warped_mask == 0] = 0.0
        canvas += warped.astype(np.float32) * layer_weight[..., None]
        weights += layer_weight

    covered = weights > 0
    canvas[covered] /= weights[covered, None]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), np.clip(canvas, 0, 255).astype(np.uint8))
    print(
        f"Rendered {len(selected)} capture groups to "
        f"{output_size[0]}x{output_size[1]}: {args.output}"
    )


if __name__ == "__main__":
    main()
