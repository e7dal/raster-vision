from typing import Tuple

import numpy as np
from PIL import ImageColor


def color_to_triple(color: str) -> Tuple[int, int, int]:
    """Given a PIL ImageColor string, return a triple of integers
    representing the red, green, and blue values.

    Args:
         color: A PIL ImageColor string

    Returns:
         An triple of integers

    """
    if color is None:
        r = np.random.randint(0, 0x100)
        g = np.random.randint(0, 0x100)
        b = np.random.randint(0, 0x100)
        return (r, g, b)
    else:
        return ImageColor.getrgb(color)


def color_to_integer(color: str) -> int:
    """Given a PIL ImageColor string, return a packed integer.

    Args:
         color: A PIL ImageColor string

    Returns:
         An integer containing the packed RGB values.

    """
    triple = color_to_triple(color)
    r = triple[0] * (1 << 16)
    g = triple[1] * (1 << 8)
    b = triple[2] * (1 << 0)
    integer = r + g + b
    return integer


def rgb_to_int_array(rgb_array):
    r = np.array(rgb_array[:, :, 0], dtype=np.uint32) * (1 << 16)
    g = np.array(rgb_array[:, :, 1], dtype=np.uint32) * (1 << 8)
    b = np.array(rgb_array[:, :, 2], dtype=np.uint32) * (1 << 0)
    return r + g + b


def boxes_to_geojson(boxes, class_ids, crs_transformer, class_map,
                     scores=None):
    """Convert boxes and associated data into a GeoJSON dict.

    Args:
        boxes: list of Box in pixel row/col format.
        class_ids: list of int (one for each box)
        crs_transformer: CRSTransformer used to convert pixel coords to map
            coords in the GeoJSON
        class_map: ClassMap used to infer class_name from class_id
        scores: optional list of score or scores.
                If floats (one for each box), property name will be "score".
                If lists of floats, property name will be "scores".

    Returns:
        dict in GeoJSON format
    """
    features = []
    for box_ind, box in enumerate(boxes):
        polygon = box.geojson_coordinates()
        polygon = [list(crs_transformer.pixel_to_map(p)) for p in polygon]

        class_id = int(class_ids[box_ind])
        class_name = class_map.get_by_id(class_id).name

        feature = {
            'type': 'Feature',
            'geometry': {
                'type': 'Polygon',
                'coordinates': [polygon]
            },
            'properties': {
                'class_id': class_id,
                'class_name': class_name
            }
        }

        if scores is not None:
            box_scores = scores[box_ind]

            if box_scores is not None:
                if type(box_scores) is list:
                    feature['properties']['scores'] = box_scores
                else:
                    feature['properties']['score'] = box_scores

        features.append(feature)

    return {'type': 'FeatureCollection', 'features': features}
