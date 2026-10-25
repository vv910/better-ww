"""Map-only dragging that preserves the configured background input backend."""

import math
import sys
import time
from numbers import Real

import win32con
from ok import TaskDisabledException
from ok.device.interaction_methods.post_message import PostMessageInteraction


_MISSING = object()


def _point(value, width, height):
    try:
        coordinates = tuple(value)
    except TypeError:
        raise ValueError('Map drag endpoints must be pixel coordinate pairs') from None
    if (len(coordinates) != 2 or any(isinstance(v, bool) or not isinstance(v, Real)
                                    or not math.isfinite(v) for v in coordinates)):
        raise ValueError('Map drag endpoints must be finite pixel coordinate pairs')
    x, y = map(round, coordinates)
    if not (0 <= coordinates[0] < width and 0 <= coordinates[1] < height
            and 0 <= x < width and 0 <= y < height):
        raise ValueError('Map drag endpoints must stay inside the capture')
    return x, y


def _finish(task, interaction, point, background, primary_error):
    errors = []
    if background:
        try:
            # Use the backend's capture -> target-window coordinate conversion.
            # Its mouse_up() currently uses an unrelated (0, 0) mouse_pos field.
            position = interaction.update_mouse_pos(*point)
            interaction.post(win32con.WM_LBUTTONUP, 0, position)
        except BaseException as error:
            errors.append(error)
            try:
                task.mouse_up()
            except BaseException as error:
                errors.append(error)
    else:
        try:
            task.mouse_up()
        except BaseException as error:
            errors.append(error)
    try:
        task.executor.reset_scene()
    except BaseException as error:
        errors.append(error)
    if not errors:
        return
    # A cleanup failure must not hide the original operation failure, and a
    # cancellation during cleanup must remain recognizable to the task runner.
    cancelled = next((error for error in errors if isinstance(error, TaskDisabledException)), None)
    if primary_error is not None:
        for error in errors:
            primary_error.add_note(f'Map drag cleanup failed: {type(error).__name__}: {error}')
        if cancelled is not None and not isinstance(primary_error, TaskDisabledException):
            raise cancelled from primary_error
    else:
        failure = cancelled or errors[0]
        for error in errors:
            if error is not failure:
                failure.add_note(f'Additional map drag cleanup failure: {type(error).__name__}: {error}')
        raise failure


def drag_route_map(task, from_xy, to_xy, duration=.3):
    """Drag within the capture in seconds; always release the button on exit.

    PostMessage needs MK_LBUTTON on every move and an explicitly positioned UP.
    Other configured backends keep their own swipe implementation.
    """
    if (isinstance(duration, bool) or not isinstance(duration, Real)
            or not math.isfinite(duration) or not 0 < duration <= 2):
        raise ValueError('Map drag duration must be greater than zero and at most two seconds')
    width, height = task.width, task.height
    if (any(isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v) or v <= 0
            for v in (width, height))):
        raise ValueError('Map drag requires a valid capture size')
    start, end = _point(from_xy, width, height), _point(to_xy, width, height)
    if start == end:
        raise ValueError('Map drag must move at least one pixel')
    interaction = task.executor.interaction
    background = isinstance(interaction, PostMessageInteraction)
    last_point = start
    previous_interval = getattr(task, 'sleep_check_interval', _MISSING)
    # Holding DOWN must not invoke combat/monthly-card logic or capture a frame.
    # Executor.sleep still handles cancellation/pause, with its visual hooks off.
    task.sleep_check_interval = -1
    try:
        if not background:
            task.swipe(*start, *end, duration=duration, after_sleep=0)
            return
        task.executor.check_enabled(check_pause=False)
        interaction.move(*start)
        task.mouse_down(*start)
        began = time.monotonic()
        steps = max(2, math.ceil(duration * 60))
        for step in range(1, steps + 1):
            remaining = began + duration * step / steps - time.monotonic()
            if remaining > 0:
                task.executor.sleep(remaining)
            # Executor's debug-mode sleep skips its own enable checks; keep
            # cancellation effective in that mode as well, before the next MOVE.
            task.executor.check_enabled(check_pause=False)
            point = tuple(round(a + (b - a) * step / steps) for a, b in zip(start, end))
            # Set before move: even a partially failed send releases at its
            # attempted cursor position rather than at a stale start point.
            last_point = point
            interaction.move(*point, down_btn=win32con.MK_LBUTTON)
    finally:
        try:
            _finish(task, interaction, last_point, background, sys.exc_info()[1])
        finally:
            if previous_interval is _MISSING:
                del task.sleep_check_interval
            else:
                task.sleep_check_interval = previous_interval
