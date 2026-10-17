import json
import math
import time
from pathlib import Path

import cv2
from ok import Box, TaskDisabledException, get_path_relative_to_exe, og

from src import text_white_color
from src.pathing.EchoRoute import EchoRoute, number, read_map_image, route_files, target_angle
from src.pathing.FallGuard import FallGuard
from src.pathing.MapLocalizer import MapLocalizer
from src.pathing.MapPlayer import find_map_player
from src.pathing.MinimapHeading import MinimapHeading
from src.pathing.MovementWindow import MovementWindow
from src.pathing.RouteMarkers import RouteMarkerDetector
from src.pathing.RouteSteering import RouteSteering
from src.pathing.RouteMapNavigation import RouteMapNavigator, capture_map_views, map_region
from src.task.BaseCombatTask import BaseCombatTask, CharDeadException
from src.task.FarmMapTask import BigMap, sort_stars
from src.task.WWOneTimeTask import WWOneTimeTask


ROUTE_DESCRIPTION = 'Follow an ordered local route, fight enemies and absorb Echoes.'
RECORD_DESCRIPTION = 'Mark a route with stars. Use the nearest visible beacon as its starting point.'


class _AirborneDuringPickup(Exception):
    pass


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
            'Keep Camera Behind Player': True,
            'Auto Glide when Airborne': True,
            'Select Levitator Automatically': False,
            'Use Liberation': True,
            'Switch to Healer before and after Combat': True,
            'Combat Timeout': 180,
            'Pickup Timeout': 20,
        })
        self.config_description.update({
            'Route File': 'Local JSON route saved by Record 3C Route. Routes with a beacon entry teleport before departure.',
            'Route Queue': 'Optional: one route JSON per line, executed in order. Later routes require teleport metadata.',
            'Navigation Interval': 'Seconds between navigation updates. Lower values use more CPU. Range: 0.1 to 0.5.',
            'Keep Camera Behind Player': 'Recenter the camera while walking and use forward movement after turns.',
            'Auto Glide when Airborne': 'Use the selected Levitator to detect airborne motion. Open the glider once per fall; do not press Jump manually.',
            'Select Levitator Automatically': 'Optional tool-wheel preparation. Disabled by default; unavailable Levitator disables fall protection without stopping the route.',
            'Combat Timeout': 'Maximum seconds for one battle before stopping the route.',
            'Pickup Timeout': 'Maximum seconds to search for Echoes after each battle.',
        })
        self._combat_deadline = None
        self._heading_detector = None
        self._steering = None
        self._camera_aligned = False
        self._camera_recenter_at = 0
        self._fall_protection_ready = False
        self._fall_guard = FallGuard()
        self._route_keys = ()
        self._last_good_minimap = None
        self._last_route_position = None
        self._active_route = None
        self._airborne_stage = False
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
        if any(route.map_layer for route in routes):
            raise ValueError('Automatic layer switching is not supported yet / 本版尚未支持自动切层，请使用同层路线')
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
                self._active_route = route
                self.log_info(f'3C route {index + 1}/{len(routes)}: {paths[index]}')
                self.localizer = MapLocalizer(read_map_image(route.map_image), route.map_region)
                self._last_good_minimap = None
                self._last_route_position = None
                if route.teleport is not None:
                    self.teleport_to_route(route)
                self.localizer.reset()
                self._fall_guard.reset()
                position = self.wait_until(self.locate, time_out=5, raise_if_not_found=False)
                if position is None or route.start.distance_to(position.xy) > route.start.tolerance:
                    position = self.recover_route_position()
                if position is None or route.start.distance_to(position.xy) > route.start.tolerance:
                    self.save_localization_diagnostic('start', expected=route.start)
                    raise RuntimeError('Cannot confirm route start / 无法确认路线起点，请检查信标落点、地图楼层和北向固定设置')
                self.prepare_fall_protection()
                self.execute_route(route)
                self.info_incr('Completed Routes')
            self.log_info('3C route queue completed / 3C 路线队列已完成', notify=True)
        except CharDeadException:
            self.log_error('Route stopped after character death / 角色死亡，路线已停止', notify=True)
            raise
        finally:
            self.stop_route_movement()
            self._active_route = None
            cv2.setNumThreads(previous_threads)

    def prepare_route_runtime(self):
        self.log_info('Preparing 3C vision and Echo model / 正在准备3C识别与声骸模型，请稍候')
        for name in ('arrow', 'box_minimap', 'char_1_text', 'char_2_text', 'char_3_text',
                     *self.get_target_names(), 'pick_up_f_hcenter_vcenter',
                     'map_way_point', 'map_way_point_big', 'gray_teleport',
                     'edge_levitator', 'wheel_levitator'):
            self.get_feature_by_name(name)
            self.sleep(0.01)  # Yield to cancellation while templates load, before movement starts.
        self._heading_detector = MinimapHeading(self.get_feature_by_name('arrow').mat)
        self.load_chars()
        # Compile and run the first inference while stationary, not during the first pickup.
        og.my_app.yolo_detect(self.frame, threshold=0.6, label=0)
        self.log_info('3C vision ready / 3C识别准备完成')

    def prepare_fall_protection(self):
        self._fall_protection_ready = False
        self._fall_guard.reset()
        self._airborne_stage = False
        if not self.config.get('Auto Glide when Airborne', True):
            return
        self.info_set('Fall Protection', 'Checking selected tool / 检查当前工具')
        selected = self.find_one('edge_levitator', threshold=0.6)
        if not selected and self.config.get('Select Levitator Automatically', False):
            self.info_set('3C Stage', 'Selecting Levitator / 正在选择控物以启用防坠落')
            try:
                selected = self.ensure_levitator(force=True)
            except TaskDisabledException:
                raise
            except Exception as error:
                self.log_warning(f'Levitator preparation failed / 控物准备失败: {error}')
                selected = False
        if not selected:
            self.info_set('Fall Protection', 'Unavailable / 未启用：请预先选择控物')
            self.log_warning('Fall protection unavailable: select Levitator before starting / 防坠落未启用，请在开始前选择控物；路线继续运行')
            return
        self.next_frame()
        brightness = self.calculate_color_percentage(text_white_color, self.get_box_by_name('edge_levitator'))
        if not self.in_team()[0] or brightness < 0.1:
            self.info_set('Fall Protection', 'Unavailable / 未启用：未确认地面状态')
            self.log_warning('Cannot confirm grounded Levitator / 无法确认控物地面状态，本条路线不启用防坠落')
            return
        self._fall_protection_ready = True
        self.info_set('Fall Protection', 'Ready / 已启用')

    def observe_route_airborne(self):
        if not self._fall_protection_ready:
            return False
        airborne = None
        if self.in_team()[0]:
            # Same established airborne cue as BaseChar.flying(): Levitator is disabled in air.
            brightness = self.calculate_color_percentage(text_white_color, self.get_box_by_name('edge_levitator'))
            airborne = bool(brightness < 0.1)
        if self._fall_guard.observe(airborne, time.monotonic()):
            self.log_info('Airborne: deploy glider once / 已确认离地，打开滑翔翼')
            self.send_key(self.key_config.get('Jump Key', 'space'), down_time=0.05)
            self.info_incr('Glider Deployments')
        active = self._fall_guard.active
        if active and not self._airborne_stage:
            self.info_set('3C Stage', 'Airborne Protection / 离地保护')
        elif not active and self._airborne_stage:
            self.info_set('3C Stage', 'Moving / 移动')
        self._airborne_stage = active
        return active

    def check_pickup_airborne(self):
        if self.observe_route_airborne():
            raise _AirborneDuringPickup()

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
        """Align map bookmarks (or a legacy Boss anchor) before selecting the beacon."""
        self.stop_route_movement()
        self.reset_to_false('3C route teleport')
        self.log_info(f'Teleport to 3C route / 传送到3C路线: {route.name}, anchor={route.teleport.anchor}')
        if route.teleport.anchor:
            self.open_anchor_map(route.teleport.anchor)
        else:
            self.send_key(self.key_config.get('Map Key', 'm'), after_sleep=.5)
            if not self.wait_feature('map_my_location', time_out=3, raise_if_not_found=False):
                raise RuntimeError('Cannot open map for route entry / 无法打开路线传送地图')
        if route.teleport.views:
            beacon = RouteMapNavigator(self, route).find(search=True)
            if beacon is None:
                self.screenshot('3c-teleport-bookmark-mismatch')
                raise RuntimeError('Cannot find route map bookmark / 未找到路线地图书签，请检查所在区域、楼层或重新录制概览')
        else:
            beacon = self.find_legacy_route_beacon(route)
        self.travel_to_route_beacon(beacon)

    def open_anchor_map(self, anchor):
        self.openF2Book('gray_book_all_monsters')
        self.click(0.13, 0.14, after_sleep=0.3)
        self.click(0.20, 0.14)
        self.send_key_down('ctrl')
        try:
            self.send_key('a')
        finally:
            self.send_key_up('ctrl')
        self.send_key('backspace')
        self.input_text(anchor)
        self.send_key('enter', after_sleep=0.5)
        self.click(0.13, 0.24, after_sleep=0.3)
        # As in the existing revive flow, click Detect once; later clicks could select a map icon.
        self.click(0.89, 0.92, after_sleep=1)
        opened = self.wait_until(lambda: self.find_feature(
            ['map_way_point', 'map_way_point_big'], box=self.box_of_screen(0.28, 0.12, 0.87, 0.90), threshold=0.6),
            time_out=6, raise_if_not_found=False)
        if not opened:
            raise RuntimeError('Cannot open route anchor map / 无法打开锚点地图，请检查图鉴中的Boss名称')

    def find_legacy_route_beacon(self, route):
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
        return beacon

    def travel_to_route_beacon(self, beacon):
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

    def recover_route_position(self):
        """Stop, visually locate the player on the full map, then re-confirm the minimap."""
        if (getattr(self, '_active_route', None) is None or self._fall_guard.active
                or self.route_enemy_visible() or not self.in_team_and_world()):
            return None
        self.stop_route_movement()
        self.info_set('3C Stage', 'Relocalizing from Map / 停步打开大地图重新定位')
        self.info_incr('Map Relocalizations')
        position = None
        try:
            self.send_key(self.key_config.get('Map Key', 'm'), after_sleep=.4)
            button = self.wait_feature('map_my_location', time_out=2, raise_if_not_found=False)
            if button:
                self.click(button, after_sleep=.3)
                self.next_frame()
                player = find_map_player(self.frame, self.get_feature_by_name('arrow').mat, map_region(self.frame))
                if player is not None:
                    position = self.localizer.locate_map_player(self.frame, player, map_region(self.frame))
            if position is None:
                self.screenshot('3c-map-relocalize-failed')
                self.log_warning('Map relocalization was not confirmed / 大地图重定位未通过校验，停止盲走')
        finally:
            # Close only the map opened by this recovery; cancellation still propagates.
            if not self.in_team()[0]:
                self.send_key('esc')
        self.wait_in_team_and_world(time_out=5)
        if position is None:
            return None
        self.localizer.reset(position_hint=position)
        self.next_frame()
        confirmed = self.wait_until(self.locate, time_out=3, raise_if_not_found=False)
        if confirmed is not None:
            self.log_info('Minimap reacquired after map relocalization / 大地图重定位后已重新确认小地图')
        return confirmed

    def locate(self):
        minimap = self.get_box_by_name('box_minimap').crop_frame(self.frame)
        position = self.localizer.locate(minimap)
        if position is not None:
            self._last_good_minimap = minimap.copy()
            self._last_route_position = position
        return position

    def save_localization_diagnostic(self, stage, expected=None):
        minimap = self.get_box_by_name('box_minimap').crop_frame(self.frame)
        expected_start = None if expected is None else (expected.x, expected.y, expected.tolerance)
        self.log_warning(f'3C localization failed: stage={stage}, capture={self.frame.shape[1]}x{self.frame.shape[0]}, '
                         f'expected_start={expected_start}, last_position={self._last_route_position}, '
                         f'diagnostic={self.localizer.last_diagnostic}')
        self.screenshot(f'3c-location-{stage}-full')
        self.screenshot(f'3c-location-{stage}-minimap', frame=minimap)
        if self._last_good_minimap is not None:
            self.screenshot(f'3c-location-{stage}-last-good', frame=self._last_good_minimap)

    def stop_route_movement(self):
        self._route_keys = ()
        self._steering = None
        self._camera_aligned = False
        for key in ('w', 'a', 's', 'd', 'space'):
            self.send_key_up(key)
        self.mouse_up(key='right')

    def pause_route_movement(self):
        # A brief visual pause releases input without forgetting the calibrated camera.
        steering = self._steering
        aligned = self._camera_aligned
        self.stop_route_movement()
        self._camera_aligned = aligned
        if steering is not None:
            self._steering = RouteSteering(steering.camera_heading)

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
        self.info_set('3C Stage', 'Moving / 移动')
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
        map_recoveries = 0
        alignment_attempts = 0
        movement = MovementWindow(min_displacement=max(3, min(6, point.tolerance * .3)))
        try:
            while time.monotonic() < deadline:
                cycle_start = time.monotonic()
                self.next_frame()
                airborne = self.observe_route_airborne()
                if not airborne and cycle_start >= enemy_check_at and self.route_enemy_visible():
                    self.stop_route_movement()
                    battles += 1
                    if battles > 5:
                        raise RuntimeError('Too many repeated battles at the same route point')
                    battle_start = time.monotonic()
                    self.fight_and_pickup()
                    deadline += time.monotonic() - battle_start
                    progress_at = time.monotonic()
                    best_distance = math.inf
                    movement.reset()
                    lost_since = None
                    self.info_set('3C Stage', 'Moving / 移动')
                    continue
                if cycle_start >= enemy_check_at:
                    enemy_check_at = cycle_start + 0.4
                if not airborne and not self._camera_aligned:
                    self.center_camera()
                    self.next_frame()
                    self._camera_aligned = True
                    self._camera_recenter_at = time.monotonic() + 1.5
                    if self.route_enemy_visible():
                        continue
                if airborne and not self._camera_aligned:
                    self.pause_route_movement()
                    self.sleep(0.1)
                    continue
                position = self.locate()
                if position is None:
                    movement.reset()
                    self.pause_route_movement()
                    if lost_since is None:
                        lost_since = time.monotonic()
                    if not airborne and map_recoveries < 1 and time.monotonic() - lost_since >= 1.5:
                        map_recoveries += 1
                        if self.recover_route_position() is not None:
                            lost_since = None
                            progress_at = time.monotonic()
                            continue
                    if time.monotonic() - lost_since >= 5:
                        self.save_localization_diagnostic('lost')
                        raise RuntimeError('Minimap position lost for 5 seconds; route stopped')
                    self.sleep(0.1)
                    continue
                if lost_since is not None and time.monotonic() - lost_since > 1:
                    self.stop_route_movement()  # A long pause can let automatic camera recentering finish.
                    lost_since = None
                    continue
                lost_since = None
                distance = point.distance_to(position.xy)
                if airborne:
                    movement.reset()
                else:
                    movement.observe(position.xy, time.monotonic())
                self.info_set('Distance to Route Point', round(distance, 1))
                if distance <= point.tolerance:
                    if not airborne:
                        reached = True
                        return
                    self.pause_route_movement()
                    self.sleep(0.1)
                    continue
                if distance < best_distance - 2:
                    best_distance = distance
                    progress_at = time.monotonic()
                elif not airborne and time.monotonic() - progress_at >= 8:
                    if movement.is_stuck(time.monotonic()):
                        if recoveries >= 2:
                            self.save_navigation_diagnostic(point, position, 'blocked', movement.displacement)
                            raise RuntimeError('Route blocked after two recovery attempts / 确认角色停滞，请检查点序、障碍和楼层')
                        self.recover_stuck(recoveries)
                        recoveries += 1
                        progress_at = time.monotonic()
                        best_distance = math.inf
                        movement.reset()
                        continue
                    if time.monotonic() - progress_at >= 16 and alignment_attempts < 1:
                        # Distance can temporarily increase while physically moving along terrain.
                        # Recheck the view/position once; do not misclassify this as a jump-worthy trap.
                        self.stop_route_movement()
                        self.recover_route_position()
                        alignment_attempts += 1
                        progress_at = time.monotonic()
                        best_distance = math.inf
                        movement.reset()
                        continue
                    if alignment_attempts and time.monotonic() - progress_at >= 30:
                        self.save_navigation_diagnostic(point, position, 'not-approaching', movement.displacement)
                        raise RuntimeError('Moving without approaching route point / 角色在移动但未接近目标，请检查误标点并补充途经点')
                heading = self.confirm_route_heading()
                if heading is None:  # Combat started while waiting for a clear arrow frame.
                    continue
                heading = self.maintain_route_camera(heading, time.monotonic()) if not airborne else heading
                if heading is None:
                    continue
                if self._steering is None:
                    self._steering = RouteSteering(heading)
                keys = self._steering.update(target_angle(position.xy, point), heading, time.monotonic())
                self.set_route_keys(keys)
                elapsed = time.monotonic() - cycle_start
                samples += 1
                vision_seconds += elapsed
                interval = min(self.navigation_interval, 0.1) if airborne else self.navigation_interval
                self.sleep(max(0.01, interval - elapsed))
            raise RuntimeError('Route waypoint timed out')
        finally:
            if not (reached and keep_moving):
                self.stop_route_movement()
            if samples:
                self.log_debug(f'3C navigation: cycles={samples}, average_cycle_ms={vision_seconds / samples * 1000:.1f}')

    def maintain_route_camera(self, heading, now):
        if (not self.config.get('Keep Camera Behind Player', True) or not self._route_keys
                or now < self._camera_recenter_at):
            return heading
        turning = self._route_keys != ('w',)
        if turning:
            # Sideways input must be released before camera coordinates change.
            self.set_route_keys(())
        self.click(0.5, 0.5, key='middle', down_time=0.02)
        self._camera_recenter_at = now + 1.5
        if turning:
            self.sleep(0.15)
            self.next_frame()
            if self.observe_route_airborne():
                return None
            if self.route_enemy_visible():
                return None
            heading = self.confirm_route_heading()
            if heading is None:
                return None
        # Middle click puts the view behind the player. Rebase steering in the new camera frame.
        self._steering = RouteSteering(heading)
        return heading

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
            self.pause_route_movement()
            self.log_debug(f'3C heading retry {attempt + 1}/3: {result}')
            if attempt < 2:
                self.sleep(0.2)
                self.next_frame()
                if self.observe_route_airborne():
                    return None
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
        for key, duration in (('s', .3), ('a' if attempt % 2 == 0 else 'd', .4)):
            self.send_key(key, down_time=duration, after_sleep=.1)
            self.next_frame()
            if self.observe_route_airborne() or self.route_enemy_visible():
                return
        # Disabled fall protection means the airborne signal is unknown, not grounded.
        if (not self._fall_protection_ready or not self.in_team()[0]
                or self.calculate_color_percentage(text_white_color, self.get_box_by_name('edge_levitator')) < .1):
            return
        self.send_key(self.key_config.get('Jump Key', 'space'), down_time=.05, after_sleep=.3)
        self.next_frame()
        self.observe_route_airborne()

    def save_navigation_diagnostic(self, point, position, reason, displacement):
        self.log_warning(f'3C navigation {reason}: target=({point.x}, {point.y}), '
                         f'position={position}, window_displacement={displacement:.2f}, '
                         f'camera_aligned={self._camera_aligned}, keys={self._route_keys}')
        self.screenshot(f'3c-navigation-{reason}')

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
        self.info_set('3C Stage', 'Combat / 战斗')
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
        empty_views = 0
        needs_alignment = True
        self.info_set('3C Stage', 'Searching Echoes / 搜索声骸')
        self.stop_route_movement()
        try:
            while time.monotonic() < deadline:
                self.next_frame()
                if self.observe_route_airborne():
                    self.pause_route_movement()
                    self.sleep(0.1)
                    continue
                if self.pick_echo():
                    self.incr_drop(True)
                    empty_views = 0
                    needs_alignment = True
                    self.sleep(0.5)
                    continue
                if self.route_enemy_visible():
                    return
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return
                if needs_alignment:
                    if remaining < 0.5:
                        return
                    self.click(0.5, 0.5, key='middle', down_time=0.02)
                    self.sleep(0.4)
                    self.next_frame()
                    needs_alignment = False
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return
                self.info_set('Echo Search View', empty_views + 1)
                # The shared turn=True helper waits >1s on every view. Sweep explicitly here.
                try:
                    dropped, _ = self.yolo_find_echo(turn=False, use_color=False, time_out=min(5, remaining),
                                                      update_function=self.check_pickup_airborne)
                except _AirborneDuringPickup:
                    self.stop_route_movement()
                    needs_alignment = True
                    continue
                if not dropped:
                    empty_views += 1
                    if empty_views >= 4 or deadline - time.monotonic() < 0.6:
                        return
                    self.send_key('a', down_time=0.05)
                    needs_alignment = True
                    continue
                self.incr_drop(True)
                empty_views = 0
                needs_alignment = True
                self.sleep(0.5)
        finally:
            self.stop_route_movement()
            # Combat/pickup can move the player and rotate the camera. Re-anchor before resuming.
            self.localizer.reset()


class Record3CEchoRouteTask(BigMap):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = '📍 Record 3C Route from Map'
        self.description = RECORD_DESCRIPTION
        self.default_config.update({'Route File': 'routes/3c/local-route.json', 'Route Name': 'Local 3C Route',
                                    'Start Method': 'Nearest Beacon', 'Capture Teleport Overview': True,
                                    'Teleport Anchor': '',
                                    'Maximum Marker Gap': 200, 'Recording Radius': 500})
        self.config_type['Start Method'] = {'type': 'drop_down', 'options': ['Nearest Beacon', 'Diamond']}
        self.config_description.update({
            'Route File': 'Save a new JSON route and map image. Existing files are never overwritten.',
            'Route Name': 'Name of this local route.',
            'Start Method': 'Nearest Beacon needs no diamond. Center the desired star group on the map; check the selected beacon and point order in the preview.',
            'Capture Teleport Overview': 'After reading stars, zoom out and save verified map views for automatic beacon search. Original route coordinates stay unchanged.',
            'Teleport Anchor': 'Optional: nearby fixed Boss name in your game language. Keep the start within 50 map pixels of an unlocked beacon.',
            'Maximum Marker Gap': 'Maximum distance between consecutive stars, in 1080p pixels. Distant groups are excluded. Reduce this to separate nearby routes.',
            'Recording Radius': 'Only include stars within this radius of the start, in 1080p pixels. Check the saved preview before farming.',
        })

    def validate_config(self, key, value):
        if key == 'Start Method' and value not in ('Nearest Beacon', 'Diamond'):
            return 'Start Method must be Nearest Beacon or Diamond'
        if key in ('Maximum Marker Gap', 'Recording Radius'):
            try:
                number(value, key, 10, 2000)
            except ValueError as error:
                return str(error)

    @staticmethod
    def recording_region(frame):
        height, width = frame.shape[:2]
        return [round(width * 0.28), round(height * 0.12), round(width * 0.87), round(height * 0.90)]

    def load_stars(self, wait_world=True, max_distance=None):
        # Select stars on the untouched user view. Optional zoom-out capture happens
        # only after the route coordinates and start beacon have been fixed.
        method = self.config.get('Start Method', 'Nearest Beacon')
        if self.validate_config('Start Method', method):
            raise ValueError('Unknown Start Method')
        self.reset()
        self.excluded_stars = []
        self._entry_beacon = None
        self._record_map_open = False
        self.next_frame()
        self.big_map_frame = self.frame.copy()
        region = self.recording_region(self.big_map_frame)
        detector = RouteMarkerDetector(self.get_feature_by_name('big_map_diamond').mat,
                                       self.get_feature_by_name('big_map_star').mat)
        diamonds, stars = detector.detect(self.big_map_frame, region)
        def convert(match):
            return Box(round(match.x), round(match.y), round(match.width), round(match.height),
                       confidence=match.confidence)
        self.log_info(f'3C map markers: capture={self.big_map_frame.shape[1]}x{self.big_map_frame.shape[0]}, '
                      f'diamonds={len(diamonds)}, stars={len(stars)}, scores={detector.best_scores}')
        self.log_info(f'3C marker candidates: region={region}, '
                      f'stars={[(tuple(round(v, 1) for v in star.center()), round(float(star.confidence), 3)) for star in stars]}, '
                      f'template_sizes=diamond:{detector.diamond_template.shape[:2]},star:{detector.star_template.shape[:2]}')
        scale = self.big_map_frame.shape[0] / 1080
        gap = number(self.config.get('Maximum Marker Gap', 200), 'Maximum Marker Gap', 10, 2000) * scale
        radius = number(self.config.get('Recording Radius', 500), 'Recording Radius', 10, 2000) * scale
        if max_distance is not None:
            gap = number(max_distance, 'max_distance', 0)
        candidates = [convert(star) for star in stars]
        if method == 'Nearest Beacon':
            left, top, right, bottom = region
            beacons = self.find_feature(['map_way_point', 'map_way_point_big'], frame=self.big_map_frame,
                                        box=Box(left, top, right - left, bottom - top), threshold=.65)
            if not beacons or not candidates:
                self.screenshot('3c-record-beacon-missing', frame=self.big_map_frame)
                raise ValueError('Need visible stars and an unlocked beacon / 地图中央需要有星标和已解锁信标，无需放置菱形')
            center = ((left + right) / 2, (top + bottom) / 2)
            seed = min(candidates, key=lambda star: math.dist(star.center(), center))
            # Select a connected group around the central star before choosing its entry.
            group, pending = [seed], [seed]
            while pending:
                current = pending.pop()
                for star in candidates:
                    if (star not in group and star.center_distance(seed) <= radius
                            and (gap == 0 or star.center_distance(current) <= gap)):
                        group.append(star)
                        pending.append(star)
            self._entry_beacon = min(beacons, key=lambda beacon: min(beacon.center_distance(star) for star in group))
            self.diamond = self._entry_beacon  # Legacy preview/start field; it now represents the real beacon.
            nearby = [star for star in group if star.center_distance(self.diamond) <= radius]
        else:
            if len(diamonds) != 1:
                self.screenshot('3c-record-marker-failed', frame=self.big_map_frame)
                if diamonds:
                    raise ValueError('Multiple start diamonds found / 发现多个起点菱形，请调整地图视野，只保留当前路线的一个菱形')
                raise ValueError('Start diamond not found / 未识别到起点菱形，请移开遮挡并将菱形放在中央地图区域')
            self.diamond = convert(diamonds[0])
            nearby = [star for star in candidates if star.center_distance(self.diamond) <= radius]
        if method == 'Nearest Beacon' and nearby:
            # The entry leg is bounded by Recording Radius. Marker Gap only separates
            # star groups; a real beacon need not sit within one star-to-star step.
            first = min(nearby, key=lambda star: star.center_distance(self.diamond))
            selected = [first, *sort_stars([star for star in nearby if star is not first], first, gap)]
        else:
            selected = sort_stars(nearby, self.diamond, gap)
        self.stars = [self.diamond, *selected]
        self.excluded_stars = [star for star in candidates if star not in selected]
        self.info_set('Selected Route Stars', len(selected))
        self.info_set('Excluded Route Stars', len(self.excluded_stars))
        self.screenshot('3c-record-markers', frame=self.render_recording_preview())
        if len(selected) < (1 if method == 'Nearest Beacon' else 2):
            if method == 'Nearest Beacon':
                raise ValueError('No reachable route stars / 信标到星标超过录制半径，请调整视野或补充途经点')
            raise ValueError('Need at least two nearby stars / 当前范围内需至少两个相连星标，请检查相邻距离与录制半径；其他标记类型不会加入路线')
        self.log_info(f'3C route selected {len(selected)} stars, excluded {len(self.excluded_stars)}, '
                      f'max_gap={gap:.0f}, radius={radius:.0f} / 请检查路线预览中的点序')
        self._record_map_open = bool(self._entry_beacon and self.config.get('Capture Teleport Overview', True))
        if not self._record_map_open or wait_world:
            self.send_key('esc')
            self._record_map_open = False
        if wait_world:
            self.wait_in_team_and_world()

    def render_recording_preview(self):
        preview = self.big_map_frame.copy()
        left, top, right, bottom = self.recording_region(preview)
        cv2.rectangle(preview, (left, top), (right, bottom), (210, 130, 40), 2)
        for star in getattr(self, 'excluded_stars', []):
            center = tuple(round(value) for value in star.center())
            cv2.drawMarker(preview, center, (60, 60, 240), cv2.MARKER_TILTED_CROSS, 24, 2)
        previous = None
        for index, star in enumerate(self.stars):
            center = tuple(round(value) for value in star.center())
            if previous is not None:
                cv2.line(preview, previous, center, (80, 230, 80), 2)
            cv2.circle(preview, center, max(12, round(star.width * 0.7)), (80, 230, 80), 2)
            cv2.putText(preview, 'S' if index == 0 else str(index), (center[0] + 16, center[1] - 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.putText(preview, 'S' if index == 0 else str(index), (center[0] + 16, center[1] - 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 255, 100), 2, cv2.LINE_AA)
            previous = center
        return preview

    def run(self):
        path = Path(get_path_relative_to_exe(self.config.get('Route File'))).resolve()
        image_path = path.with_suffix('.map.png')
        preview_path = path.with_suffix('.preview.png')
        if path.suffix.lower() != '.json':
            raise ValueError('Route File must end in .json')
        if path.exists() or image_path.exists() or preview_path.exists():
            raise FileExistsError('Route already exists. Choose a new Route File.')
        if any(path.with_suffix(f'.view-{index}.png').exists() for index in range(1, 7)):
            raise FileExistsError('Route map overview already exists. Choose a new Route File.')
        WWOneTimeTask.run(self)
        self.load_stars(wait_world=False)
        # Exclude map-side UI from feature matching. Keep all route markers visible here.
        region = self.recording_region(self.big_map_frame)
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
        view_images = []
        if getattr(self, '_entry_beacon', None) is not None:
            data['teleport'] = {'anchor': anchor, 'x': x, 'y': y,
                                'views': [{'map_image': image_path.name, 'x': x, 'y': y, 'map_region': region}]}
            try:
                if self.config.get('Capture Teleport Overview', True):
                    self.info_set('3C Stage', 'Recording Map Overview / 正在录制传送地图概览')
                    views = capture_map_views(self, self.big_map_frame, (x, y), tuple(region))
                    for index, (frame, point, view_region) in enumerate(views, 1):
                        view_path = path.with_suffix(f'.view-{index}.png')
                        view_images.append((view_path, frame))
                        data['teleport']['views'].append({'map_image': view_path.name, 'x': point[0], 'y': point[1],
                                                          'map_region': list(view_region)})
            finally:
                if getattr(self, '_record_map_open', False):
                    self.send_key('esc')
                    self._record_map_open = False
        elif anchor:
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
        preview_encoded, preview = cv2.imencode('.png', self.render_recording_preview())
        if not encoded or not preview_encoded:
            raise ValueError('Cannot encode route map')
        with image_path.open('xb') as output:
            output.write(image.tobytes())
        with preview_path.open('xb') as output:
            output.write(preview.tobytes())
        for view_path, frame in view_images:
            success, content = cv2.imencode('.png', frame)
            if not success:
                raise ValueError('Cannot encode route overview')
            with view_path.open('xb') as output:
                output.write(content.tobytes())
        with path.open('x', encoding='utf-8') as output:
            json.dump(data, output, ensure_ascii=False, indent=2)
        self.info_set('Route File', str(path))
        self.info_set('Route Preview', str(preview_path))
        self.info_set('Route Points', len(data['waypoints']))
        self.log_info('Route saved. Check point order before farming / 路线已保存，请检查点序后开始刷取', notify=True)
