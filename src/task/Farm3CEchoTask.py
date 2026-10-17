import json
import math
import time
from pathlib import Path

import cv2
from ok import get_path_relative_to_exe

from src.pathing.EchoRoute import EchoRoute, movement_direction, number, read_map_image, target_angle
from src.pathing.MapLocalizer import MapLocalizer
from src.pathing.MinimapHeading import MinimapHeading
from src.task.BaseCombatTask import BaseCombatTask, CharDeadException
from src.task.FarmMapTask import BigMap
from src.task.WWOneTimeTask import WWOneTimeTask


ROUTE_DESCRIPTION = 'Follow an ordered local route, fight enemies and absorb Echoes.'
RECORD_DESCRIPTION = 'Start on the map with a diamond at the start and stars along the route.'


class Farm3CEchoTask(WWOneTimeTask, BaseCombatTask):
    owns_switch_healer_config = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = '🗺️ Farm 3C Echo by Route'
        self.description = ROUTE_DESCRIPTION
        self.default_config.update({
            'Route File': 'routes/3c/local-route.json',
            'Use Liberation': True,
            'Switch to Healer before and after Combat': True,
            'Combat Timeout': 180,
            'Pickup Timeout': 20,
        })
        self.config_description.update({
            'Route File': 'Local JSON route saved by Record 3C Route. Start near its diamond marker.',
            'Combat Timeout': 'Maximum seconds for one battle before stopping the route.',
            'Pickup Timeout': 'Maximum seconds to search for Echoes after each battle.',
        })
        self._combat_deadline = None
        self._heading_detector = None

    def validate_config(self, key, value):
        if key in ('Combat Timeout', 'Pickup Timeout'):
            try:
                number(value, key, 5, 600)
            except ValueError as error:
                return str(error)

    def run(self):
        route = EchoRoute.load(get_path_relative_to_exe(self.config.get('Route File')))
        self.combat_timeout = number(self.config.get('Combat Timeout', 180), 'Combat Timeout', 5, 600)
        self.pickup_timeout = number(self.config.get('Pickup Timeout', 20), 'Pickup Timeout', 5, 600)
        self.localizer = MapLocalizer(read_map_image(route.map_image), route.map_region)
        self.use_liberation = self.config.get('Use Liberation', True)
        self.start_time = time.time()
        WWOneTimeTask.run(self)
        self.wait_in_team_and_world(time_out=10)
        self.center_camera()
        position = self.wait_until(self.locate, time_out=5, raise_if_not_found=False)
        if position is None or route.start.distance_to(position.xy) > route.start.tolerance:
            raise RuntimeError('Cannot confirm route start. Stand near the recorded diamond and use a north-up minimap.')
        self.info_set('Route', route.name)
        self.info_set('Route Points', len(route.waypoints))
        try:
            for index, point in enumerate(route.waypoints):
                self.info_set('Current Route Point', index + 1)
                self.follow_point(point)
                if point.action == 'combat':
                    if self.wait_until(lambda: self.in_combat(target=True), time_out=5, raise_if_not_found=False):
                        self.fight_and_pickup()
                    else:
                        self.info_incr('Empty Combat Points')
                        self.pickup_echoes()
                elif point.action == 'pickup':
                    self.pickup_echoes()
                self.info_incr('Completed Route Points')
            self.log_info('3C route completed / 3C 路线已完成', notify=True)
        except CharDeadException:
            self.log_error('Route stopped after character death / 角色死亡，路线已停止', notify=True)
            raise
        finally:
            self.stop_route_movement()

    def locate(self):
        minimap = self.get_box_by_name('box_minimap').crop_frame(self.frame)
        return self.localizer.locate(minimap)

    def stop_route_movement(self):
        for key in ('w', 'a', 's', 'd', 'space'):
            self.send_key_up(key)
        self.mouse_up(key='right')

    def follow_point(self, point):
        # Movement is pulsed: every new heading requires a fresh visual position.
        deadline = time.monotonic() + point.timeout
        lost_since = None
        progress_at = time.monotonic()
        best_distance = math.inf
        recoveries = 0
        battles = 0
        try:
            while time.monotonic() < deadline:
                self.next_frame()
                if self.in_combat(target=True):
                    battles += 1
                    if battles > 5:
                        raise RuntimeError('Too many repeated battles at the same route point')
                    battle_start = time.monotonic()
                    self.fight_and_pickup()
                    deadline += time.monotonic() - battle_start
                    progress_at = time.monotonic()
                    best_distance = math.inf
                    lost_since = None
                    self.center_camera()
                    continue
                position = self.locate()
                if position is None:
                    if lost_since is None:
                        lost_since = time.monotonic()
                    if time.monotonic() - lost_since >= 5:
                        raise RuntimeError('Minimap position lost for 5 seconds; route stopped')
                    self.sleep(0.1)
                    continue
                lost_since = None
                distance = point.distance_to(position.xy)
                self.info_set('Distance to Route Point', round(distance, 1))
                if distance <= point.tolerance:
                    return
                if distance < best_distance - 2:
                    best_distance = distance
                    progress_at = time.monotonic()
                elif time.monotonic() - progress_at >= 8:
                    if recoveries >= 2:
                        raise RuntimeError('Route blocked after two recovery attempts')
                    self.recover_stuck(recoveries)
                    recoveries += 1
                    progress_at = time.monotonic()
                    best_distance = math.inf
                    continue
                # W/A/S/D are camera-relative; align the camera to the player before each pulse.
                self.center_camera()
                self.next_frame()
                if self.in_combat():
                    continue
                heading = self.confirm_route_heading()
                if heading is None:  # Combat started while waiting for a clear arrow frame.
                    continue
                direction = movement_direction(heading, target_angle(position.xy, point))
                self.send_key(direction, down_time=0.15 if distance < point.tolerance * 2 else 0.3,
                              after_sleep=0.05)
            raise RuntimeError('Route waypoint timed out')
        finally:
            self.stop_route_movement()

    def read_route_heading(self):
        if self._heading_detector is None:
            self._heading_detector = MinimapHeading(self.get_feature_by_name('arrow').mat)
        minimap = self.get_box_by_name('box_minimap').crop_frame(self.frame)
        return self._heading_detector.estimate(minimap)

    def confirm_route_heading(self):
        result = None
        for attempt in range(3):
            result = self.read_route_heading()
            if result is not None and result.reliable:
                return result.angle
            self.log_debug(f'3C heading retry {attempt + 1}/3: {result}')
            if attempt < 2:
                self.sleep(0.2)
                self.next_frame()
                if self.in_combat():
                    return None
        self.save_heading_diagnostic(result)
        raise RuntimeError('Cannot confirm minimap player heading after 3 frames / '
                           '连续3帧无法确认小地图朝向，已保存3c-heading-minimap截图')

    def save_heading_diagnostic(self, result):
        minimap = self.get_box_by_name('box_minimap').crop_frame(self.frame)
        self.log_warning(f'3C heading failed: capture={self.frame.shape[1]}x{self.frame.shape[0]}, '
                         f'minimap={minimap.shape[1]}x{minimap.shape[0]}, estimate={result}')
        self.screenshot('3c-heading-minimap', frame=minimap)

    def recover_stuck(self, attempt):
        self.stop_route_movement()
        self.info_incr('Route Recovery Attempts')
        self.send_key('s', down_time=0.3, after_sleep=0.1)
        self.send_key('a' if attempt % 2 == 0 else 'd', down_time=0.4, after_sleep=0.1)
        self.send_key('space', down_time=0.05, after_sleep=0.3)
        self.center_camera()

    def in_combat(self, target=False):
        # CombatCheck catches callback exceptions; enforce the deadline outside that wrapper.
        if self._combat_deadline is not None and time.monotonic() >= self._combat_deadline:
            raise RuntimeError('3C combat timed out')
        return super().in_combat(target=target)

    def revive_action(self):
        # A healing teleport would invalidate the snapshot position. Stop and restart at the route start.
        return False

    def is_open_world_auto_combat(self):
        return True

    def fight_and_pickup(self):
        self.stop_route_movement()
        self._combat_deadline = time.monotonic() + self.combat_timeout
        try:
            self.combat_once(wait_combat_time=1, raise_if_not_found=False, target=True)
            if time.monotonic() >= self._combat_deadline:
                raise RuntimeError('3C combat timed out')
        finally:
            self._combat_deadline = None
            self.stop_route_movement()
        self.pickup_echoes()

    def pickup_echoes(self):
        deadline = time.monotonic() + self.pickup_timeout
        # Several enemies can leave several Echoes. Search again after every pickup.
        try:
            while time.monotonic() < deadline:
                if self.pick_echo():
                    self.incr_drop(True)
                    self.sleep(0.5)
                    continue
                if self.in_combat():
                    return
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return
                dropped, _ = self.yolo_find_echo(turn=True, use_color=False, time_out=min(5, remaining))
                if not dropped:
                    return
                self.incr_drop(True)
                self.sleep(0.5)
        finally:
            self.stop_route_movement()


class Record3CEchoRouteTask(BigMap):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = '📍 Record 3C Route from Map'
        self.description = RECORD_DESCRIPTION
        self.default_config.update({'Route File': 'routes/3c/local-route.json', 'Route Name': 'Local 3C Route'})
        self.config_description.update({
            'Route File': 'Save a new JSON route and map image. Existing files are never overwritten.',
            'Route Name': 'Name of this local route.',
        })

    def run(self):
        path = Path(get_path_relative_to_exe(self.config.get('Route File'))).resolve()
        image_path = path.with_suffix('.map.png')
        if path.suffix.lower() != '.json':
            raise ValueError('Route File must end in .json')
        if path.exists() or image_path.exists():
            raise FileExistsError('Route already exists. Choose a new Route File.')
        WWOneTimeTask.run(self)
        self.load_stars(wait_world=False, max_distance=0)
        height, width = self.big_map_frame.shape[:2]
        # Exclude map-side UI from feature matching. Keep all route markers visible here.
        region = [round(width * 0.28), round(height * 0.12), round(width * 0.87), round(height * 0.90)]
        x, y = self.diamond.center()
        data = {
            'schema_version': 1, 'echo_cost': 3, 'name': self.config.get('Route Name'),
            'map_image': image_path.name, 'map_region': region,
            'start': {'x': x, 'y': y, 'tolerance': 50},
            'waypoints': [{'x': star.center()[0], 'y': star.center()[1], 'action': 'combat'}
                          for star in self.stars[1:]],
        }
        for point in (data['start'], *data['waypoints']):
            if not (region[0] <= point['x'] < region[2] and region[1] <= point['y'] < region[3]):
                raise ValueError('Keep the diamond and all stars in the central visible map area')
        MapLocalizer(self.big_map_frame, region)  # reject unusable snapshots before saving
        path.parent.mkdir(parents=True, exist_ok=True)
        encoded, image = cv2.imencode('.png', self.big_map_frame)
        if not encoded:
            raise ValueError('Cannot encode route map')
        with image_path.open('xb') as output:
            output.write(image.tobytes())
        with path.open('x', encoding='utf-8') as output:
            json.dump(data, output, ensure_ascii=False, indent=2)
        self.info_set('Route File', str(path))
        self.info_set('Route Points', len(data['waypoints']))
        self.log_info('Route saved. Check point order before farming / 路线已保存，请检查点序后开始刷取', notify=True)
