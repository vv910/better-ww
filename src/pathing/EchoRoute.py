import json
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


def read_map_image(path):
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'Cannot read route map image: {path}')
    return image


def number(value, field, minimum=0, maximum=100000):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{field} must be a number')
    if not math.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError(f'{field} must be between {minimum} and {maximum}')
    return float(value)


@dataclass(frozen=True)
class RoutePoint:
    x: float
    y: float
    action: str = 'walk'
    tolerance: float = 12
    timeout: float = 120

    def distance_to(self, position):
        return math.hypot(self.x - position[0], self.y - position[1])

    @classmethod
    def from_dict(cls, data, field):
        if not isinstance(data, dict):
            raise ValueError(f'{field} must be an object')
        action = data.get('action', 'walk')
        if action not in ('walk', 'combat', 'pickup'):
            raise ValueError(f'{field}.action must be walk, combat or pickup')
        return cls(number(data.get('x'), f'{field}.x'), number(data.get('y'), f'{field}.y'), action,
                   number(data.get('tolerance', 12), f'{field}.tolerance', 1, 80),
                   number(data.get('timeout', 120), f'{field}.timeout', 5, 600))


@dataclass(frozen=True)
class EchoRoute:
    name: str
    map_image: Path
    start: RoutePoint
    waypoints: tuple[RoutePoint, ...]
    map_region: tuple[int, int, int, int]
    teleport: 'RouteTeleport | None' = None

    @classmethod
    def load(cls, path):
        path = Path(path).expanduser().resolve()
        data = json.loads(path.read_text(encoding='utf-8-sig'))
        if not isinstance(data, dict) or type(data.get('schema_version')) is not int or data['schema_version'] != 1:
            raise ValueError('Route schema_version must be 1')
        if data.get('echo_cost') != 3 or isinstance(data.get('echo_cost'), bool):
            raise ValueError('Route echo_cost must be 3')
        name = data.get('name')
        if not isinstance(name, str) or not name.strip():
            raise ValueError('Route name must not be empty')
        image_name = data.get('map_image')
        if not isinstance(image_name, str) or not image_name.strip():
            raise ValueError('Route map_image must be a local image path')
        image_path = (path.parent / image_name).resolve()
        image = read_map_image(image_path)
        height, width = image.shape[:2]
        region = data.get('map_region', [0, 0, width, height])
        if (not isinstance(region, list) or len(region) != 4
                or any(type(value) is not int for value in region)):
            raise ValueError('Route map_region must contain four integer pixel coordinates')
        x1, y1, x2, y2 = region
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError('Route map_region is outside the map image')
        start = RoutePoint.from_dict(data.get('start'), 'start')
        points = data.get('waypoints')
        if not isinstance(points, list) or not 1 <= len(points) <= 2000:
            raise ValueError('Route requires 1 to 2000 ordered waypoints')
        waypoints = tuple(RoutePoint.from_dict(point, f'waypoints[{index}]') for index, point in enumerate(points))
        for point in (start, *waypoints):
            if not (x1 <= point.x < x2 and y1 <= point.y < y2):
                raise ValueError('Route point is outside map_region')
        teleport = None
        if 'teleport' in data:
            spec = data['teleport']
            if not isinstance(spec, dict) or not isinstance(spec.get('anchor'), str) or not spec['anchor'].strip():
                raise ValueError('teleport.anchor must be a nearby fixed Boss name in the game language')
            waypoint = RoutePoint.from_dict(spec, 'teleport')
            if not (x1 <= waypoint.x < x2 and y1 <= waypoint.y < y2):
                raise ValueError('Teleport waypoint is outside map_region')
            if start.distance_to((waypoint.x, waypoint.y)) > start.tolerance:
                raise ValueError('Route start must be within its tolerance of the teleport waypoint')
            teleport = RouteTeleport(spec['anchor'].strip(), waypoint)
        return cls(name, image_path, start, waypoints, tuple(region), teleport)


@dataclass(frozen=True)
class RouteTeleport:
    anchor: str
    waypoint: RoutePoint


def route_files(single, queue):
    """A multiline queue overrides the original single-file setting, preserving order."""
    if not isinstance(queue, str) or not isinstance(single, str):
        raise ValueError('Route File and Route Queue must be text')
    paths = [line.strip() for line in queue.splitlines() if line.strip()]
    if not paths:
        paths = [single.strip()]
    if not 1 <= len(paths) <= 100 or any(not path for path in paths):
        raise ValueError('Select 1 to 100 route files')
    return paths


def target_angle(position, point):
    """Clockwise heading with east at 0, matching BaseWWTask's arrow template."""
    return math.degrees(math.atan2(point.y - position[1], point.x - position[0])) % 360


def movement_direction(heading, target):
    error = (target - heading + 180) % 360 - 180
    if -45 <= error <= 45:
        return 'w'
    if 45 < error < 135:
        return 'd'
    if -135 < error < -45:
        return 'a'
    return 's'
