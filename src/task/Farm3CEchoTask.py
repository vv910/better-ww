import json
import math
import time
from pathlib import Path

import cv2
from ok import Box, get_path_relative_to_exe, og

from src.pathing.EchoRoute import EchoRoute, number, read_map_image, route_files, target_angle
from src.pathing.MapLocalizer import MapLocalizer
from src.pathing.MinimapHeading import MinimapHeading
from src.pathing.RouteSteering import RouteSteering
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
            'Route Queue': '',
            'Navigation Interval': 0.2,
            'Use Liberation': True,
            'Switch to Healer before and after Combat': True,
            'Combat Timeout': 180,
            'Pickup Timeout': 20,
        })
        self.config_description.update({
            'Route File': 'Local JSON route saved by Record 3C Route. Start near its diamond marker.',
            'Route Queue': 'Optional: one route JSON per line, executed in order. Later routes require teleport metadata.',
            'Navigation Interval': 'Seconds between navigation updates. Lower values use more CPU. Range: 0.1 to 0.5.',
            'Combat Timeout': 'Maximum seconds for one battle before stopping the route.',
            'Pickup Timeout': 'Maximum seconds to search for Echoes after each battle.',
        })
        self._combat_deadline = None
        self._heading_detector = None
        self._steering = None
        self._route_keys = ()
        self.config_type['Route Queue'] = {'type': 'text_edit'}

    def validate_config(self, key, value):
        if key in ('Combat Timeout', 'Pickup Timeout'):
            try:
                number(value, key, 5, 600)
            except ValueError as error:
                return str(error)
        elif key == 'Navigation Interval':
            try:
                number(value, key, 0.1, 0.5)
            except ValueError as error:
                return str(error)

    def run(self):
        paths = route_files(self.config.get('Route File'), self.config.get('Route Queue', ''))
        routes = [EchoRoute.load(get_path_relative_to_exe(path)) for path in paths]
        if any(route.teleport is None for route in routes[1:]):
            raise ValueError('Every route after the first needs teleport metadata / 后续路线需要配置传送信息')
        self.combat_timeout = number(self.config.get('Combat Timeout', 180), 'Combat Timeout', 5, 600)
        self.pickup_timeout = number(self.config.get('Pickup Timeout', 20), 'Pickup Timeout', 5, 600)
        self.navigation_interval = number(self.config.get('Navigation Interval', 0.2),
                                          'Navigation Interval', 0.1, 0.5)
        self.use_liberation = self.config.get('Use Liberation', True)
        self.start_time = time.time()
        WWOneTimeTask.run(self)
        self.wait_in_team_and_world(time_out=10)
        previous_threads = cv2.getNumThreads()
        try:
            # OpenCV's default worker count can contend with the game on high-core CPUs.
            cv2.setNumThreads(min(previous_threads, 2))
            self.prepare_route_runtime()
            self.info_set('Queued Routes', len(routes))
            for index, route in enumerate(routes):
                self.stop_route_movement()
                self.info_set('Current Route', index + 1)
                self.info_set('Route', route.name)
                self.log_info(f'3C route {index + 1}/{len(routes)}: {paths[index]}')
                self.localizer = MapLocalizer(read_map_image(route.map_image), route.map_region)
                if route.teleport is not None:
                    self.teleport_to_route(route)
                self.localizer.reset()
                position = self.wait_until(self.locate, time_out=5, raise_if_not_found=False)
                if position is None or route.start.distance_to(position.xy) > route.start.tolerance:
                    raise RuntimeError('Cannot confirm route start. Stand near the recorded diamond and use a north-up minimap.')
                self.execute_route(route)
                self.info_incr('Completed Routes')
            self.log_info('3C route queue completed / 3C 路线队列已完成', notify=True)
        except CharDeadException:
            self.log_error('Route stopped after character death / 角色死亡，路线已停止', notify=True)
            raise
        finally:
            self.stop_route_movement()
            cv2.setNumThreads(previous_threads)

    def prepare_route_runtime(self):
        self.log_info('Preparing 3C vision and Echo model / 正在准备3C识别与声骸模型，请稍候')
        for name in ('arrow', 'box_minimap', 'char_1_text', 'char_2_text', 'char_3_text',
                     *self.get_target_names(), 'pick_up_f_hcenter_vcenter',
                     'map_way_point', 'map_way_point_big', 'gray_teleport'):
            self.get_feature_by_name(name)
            self.sleep(0.01)  # Yield to cancellation while templates load, before movement starts.
        self._heading_detector = MinimapHeading(self.get_feature_by_name('arrow').mat)
        self.load_chars()
        # Compile and run the first inference while stationary, not during the first pickup.
        og.my_app.yolo_detect(self.frame, threshold=0.6, label=0)
        self.log_info('3C vision ready / 3C识别准备完成')

    def execute_route(self, route):
        self.info_set('Route Points', len(route.waypoints))
        try:
            for index, point in enumerate(route.waypoints):
                self.info_set('Current Route Point', index + 1)
                self.follow_point(point, keep_moving=point.action == 'walk' and index < len(route.waypoints) - 1)
                if point.action == 'combat':
                    if self.wait_until(lambda: self.in_combat(target=True), time_out=1, raise_if_not_found=False):
                        self.fight_and_pickup()
                    else:
                        self.info_incr('Empty Combat Points')
                        self.pickup_echoes()
                elif point.action == 'pickup':
                    self.pickup_echoes()
                self.info_incr('Completed Route Points')
        finally:
            self.stop_route_movement()

    def teleport_to_route(self, route):
        """Use a fixed Boss as a map anchor, then visually select the recorded beacon."""
        self.stop_route_movement()
        self.reset_to_false('3C route teleport')
        self.log_info(f'Teleport to 3C route / 传送到3C路线: {route.name}, anchor={route.teleport.anchor}')
        self.openF2Book('gray_book_all_monsters')
        self.click(0.13, 0.14, after_sleep=0.3)
        self.click(0.20, 0.14)
        self.send_key_down('ctrl')
        try:
            self.send_key('a')
        finally:
            self.send_key_up('ctrl')
        self.send_key('backspace')
        self.input_text(route.teleport.anchor)
        self.send_key('enter', after_sleep=0.5)
        self.click(0.13, 0.24, after_sleep=0.3)
        # As in the existing revive flow, click Detect once; later clicks could select a map icon.
        self.click(0.89, 0.92, after_sleep=1)
        opened = self.wait_until(lambda: self.find_feature(
            ['map_way_point', 'map_way_point_big'], box=self.box_of_screen(0.28, 0.12, 0.87, 0.90), threshold=0.6),
            time_out=6, raise_if_not_found=False)
        if not opened:
            raise RuntimeError('Cannot open route anchor map / 无法打开锚点地图，请检查图鉴中的Boss名称')
        height, width = self.frame.shape[:2]
        region = (round(width * 0.28), round(height * 0.12), round(width * 0.87), round(height * 0.90))
        waypoint = route.teleport.waypoint
        projected = self.localizer.project_map_point(self.frame, (waypoint.x, waypoint.y), region)
        if projected is None:
            self.screenshot('3c-teleport-map-mismatch')
            raise RuntimeError('Cannot align route anchor map / 锚点地图与路线截图不匹配，请选择同一区域的固定Boss')
        radius = max(20, round(width * 0.018))
        search = Box(max(0, round(projected.x) - radius), max(0, round(projected.y) - radius),
                     radius * 2, radius * 2)
        beacons = self.find_feature(['map_way_point', 'map_way_point_big'], box=search, threshold=0.65)
        if not beacons:
            self.screenshot('3c-teleport-beacon-missing')
            raise RuntimeError('Recorded teleport beacon not found / 没有找到记录的信标，请检查是否已解锁或被标记遮挡')
        beacon = min(beacons, key=lambda box: math.dist(box.center(), (projected.x, projected.y)))
        self.click(beacon, after_sleep=0.5)
        travel = self.wait_feature('gray_teleport', time_out=2, raise_if_not_found=False)
        if not travel:
            popup = self.find_feature('map_way_point', box='map_way_point_pop_up_box', threshold=0.65)
            if popup:
                self.click(popup[0], after_sleep=0.3)
                travel = self.wait_feature('gray_teleport', time_out=2, raise_if_not_found=False)
        if not travel:
            raise RuntimeError('Beacon cannot be teleported to / 信标不可传送或尚未解锁')
        self.click_box(travel, relative_x=1.5, after_sleep=0.5)
        self.wait_in_team_and_world(time_out=30)
        self.localizer.reset()
        self.next_frame()

    def locate(self):
        minimap = self.get_box_by_name('box_minimap').crop_frame(self.frame)
        return self.localizer.locate(minimap)

    def stop_route_movement(self):
        self._route_keys = ()
        self._steering = None
        for key in ('w', 'a', 's', 'd', 'space'):
            self.send_key_up(key)
        self.mouse_up(key='right')

    def set_route_keys(self, keys):
        previous = self._route_keys
        # Preserve common keys (e.g. W when changing W -> W+D), so motion stays continuous.
        for key in previous:
            if key not in keys:
                self.send_key_up(key)
        for key in keys:
            if key not in previous:
                self.send_key_down(key)
        self._route_keys = keys

    def route_enemy_visible(self):
        # Passive check only: no character recognition, camera reset or retarget wait while walking.
        return self._in_combat or self.has_health_bar()

    def follow_point(self, point, keep_moving=False):
        deadline = time.monotonic() + point.timeout
        lost_since = None
        progress_at = time.monotonic()
        best_distance = math.inf
        recoveries = 0
        battles = 0
        enemy_check_at = 0
        reached = False
        samples = 0
        vision_seconds = 0
        try:
            while time.monotonic() < deadline:
                cycle_start = time.monotonic()
                self.next_frame()
                if cycle_start >= enemy_check_at and self.route_enemy_visible():
                    self.stop_route_movement()
                    battles += 1
                    if battles > 5:
                        raise RuntimeError('Too many repeated battles at the same route point')
                    battle_start = time.monotonic()
                    self.fight_and_pickup()
                    deadline += time.monotonic() - battle_start
                    progress_at = time.monotonic()
                    best_distance = math.inf
                    lost_since = None
                    continue
                if cycle_start >= enemy_check_at:
                    enemy_check_at = cycle_start + 0.4
                position = self.locate()
                if position is None:
                    self.stop_route_movement()
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
                    reached = True
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
                if self._steering is None:
                    self.center_camera()
                    self.next_frame()
                    if self.route_enemy_visible():
                        continue
                heading = self.confirm_route_heading()
                if heading is None:  # Combat started while waiting for a clear arrow frame.
                    continue
                if self._steering is None:
                    self._steering = RouteSteering(heading)
                keys = self._steering.update(target_angle(position.xy, point), heading, time.monotonic())
                self.set_route_keys(keys)
                elapsed = time.monotonic() - cycle_start
                samples += 1
                vision_seconds += elapsed
                self.sleep(max(0.01, self.navigation_interval - elapsed))
            raise RuntimeError('Route waypoint timed out')
        finally:
            if not (reached and keep_moving):
                self.stop_route_movement()
            if samples:
                self.log_debug(f'3C navigation: cycles={samples}, average_cycle_ms={vision_seconds / samples * 1000:.1f}')

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
            steering = self._steering
            self.stop_route_movement()
            if steering is not None:
                self._steering = RouteSteering(steering.camera_heading)
            self.log_debug(f'3C heading retry {attempt + 1}/3: {result}')
            if attempt < 2:
                self.sleep(0.2)
                self.next_frame()
                if self.route_enemy_visible():
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
        self.default_config.update({'Route File': 'routes/3c/local-route.json', 'Route Name': 'Local 3C Route',
                                    'Teleport Anchor': ''})
        self.config_description.update({
            'Route File': 'Save a new JSON route and map image. Existing files are never overwritten.',
            'Route Name': 'Name of this local route.',
            'Teleport Anchor': 'Optional: nearby fixed Boss name in your game language. Keep the start within 50 map pixels of an unlocked beacon.',
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
        anchor = self.config.get('Teleport Anchor', '').strip()
        if anchor:
            left, top, right, bottom = region
            beacons = self.find_feature(['map_way_point', 'map_way_point_big'], frame=self.big_map_frame,
                                        box=Box(left, top, right - left, bottom - top), threshold=0.65)
            if not beacons:
                raise ValueError('No beacon visible. Keep an unlocked beacon near the diamond / 菱形附近需要有已解锁信标')
            beacon = min(beacons, key=lambda box: math.dist(box.center(), (x, y)))
            if math.dist(beacon.center(), (x, y)) > data['start']['tolerance']:
                raise ValueError('Place the diamond within 50 map pixels of an unlocked beacon / 起点菱形应在已解锁信标50地图像素以内')
            data['teleport'] = {'anchor': anchor, 'x': beacon.center()[0], 'y': beacon.center()[1]}
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
