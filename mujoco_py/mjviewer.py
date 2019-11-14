from threading import Lock
import glfw
from mujoco_py.builder import cymj
from mujoco_py.generated import const
import time
import copy
from multiprocessing import Process, Queue
from mujoco_py.utils import rec_copy, rec_assign
import numpy as np
import imageio


class MjViewerBasic(cymj.MjRenderContextWindow):
    """
    A simple display GUI showing the scene of an :class:`.MjSim` with a mouse-movable camera.

    :class:`.MjViewer` extends this class to provide more sophisticated playback and interaction controls.

    Parameters
    ----------
    sim : :class:`.MjSim`
        The simulator to display.
    """

    def __init__(self, sim):
        super().__init__(sim)

        self._gui_lock = Lock()
        self._button_left_pressed = False
        self._button_right_pressed = False
        self._last_mouse_x = 0
        self._last_mouse_y = 0

        framebuffer_width, _ = glfw.get_framebuffer_size(self.window)
        window_width, _ = glfw.get_window_size(self.window)
        self._scale = framebuffer_width * 1.0 / window_width

        glfw.set_cursor_pos_callback(self.window, self._cursor_pos_callback)
        glfw.set_mouse_button_callback(
            self.window, self._mouse_button_callback)
        glfw.set_scroll_callback(self.window, self._scroll_callback)
        glfw.set_key_callback(self.window, self.key_callback)

        self.exit = False
        self.restart = False

    def render(self):
        """
        Render the current simulation state to the screen or off-screen buffer.
        Call this in your main loop.
        """
        if self.window is None:
            return
        elif glfw.window_should_close(self.window):
            self.exit = True

        with self._gui_lock:
            super().render()

        glfw.poll_events()

    def key_callback(self, window, key, scancode, action, mods):
        if action == glfw.RELEASE and key == glfw.KEY_ESCAPE:
            print("Pressed ESC")
            self.exit = True

    def _cursor_pos_callback(self, window, xpos, ypos):
        if not (self._button_left_pressed or self._button_right_pressed):
            return

        # Determine whether to move, zoom or rotate view
        mod_shift = (
            glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS or
            glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS)
        if self._button_right_pressed:
            action = const.MOUSE_MOVE_H if mod_shift else const.MOUSE_MOVE_V
        elif self._button_left_pressed:
            action = const.MOUSE_ROTATE_H if mod_shift else const.MOUSE_ROTATE_V
        else:
            action = const.MOUSE_ZOOM

        # Determine
        dx = int(self._scale * xpos) - self._last_mouse_x
        dy = int(self._scale * ypos) - self._last_mouse_y
        width, height = glfw.get_framebuffer_size(window)

        with self._gui_lock:
            self.move_camera(action, dx / height, dy / height)

        self._last_mouse_x = int(self._scale * xpos)
        self._last_mouse_y = int(self._scale * ypos)

    def _mouse_button_callback(self, window, button, act, mods):
        self._button_left_pressed = (
            glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS)
        self._button_right_pressed = (
            glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS)

        x, y = glfw.get_cursor_pos(window)
        self._last_mouse_x = int(self._scale * x)
        self._last_mouse_y = int(self._scale * y)

    def _scroll_callback(self, window, x_offset, y_offset):
        with self._gui_lock:
            self.move_camera(const.MOUSE_ZOOM, 0, -0.05 * y_offset)


class MjViewer(MjViewerBasic):
    """
    Extends :class:`.MjViewerBasic` to add video recording, interactive time and interaction controls.

    Parameters
    ----------
    sim : :class:`.MjSim`
        The simulator to display.
    """

    def __init__(self, sim, display_all_text=False):
        super().__init__(sim)

        self._ncam = sim.model.ncam
        self._paused = False  # is viewer paused.

        # should we advance viewer just by one step.
        self._advance_by_one_step = False

        # Vars for recording video
        self._record_video = False
        self._video_queue = Queue()
        self._video_idx = 0
        self._video_path = "/tmp/video_%07d.mp4"

        # vars for capturing screen
        self._image_idx = 0
        self._image_path = "/tmp/frame_%07d.png"

        # run_speed = x1, means running real time, x2 means fast-forward times
        # two.
        self._run_speed = 1.0
        self._loop_count = 0
        self._render_every_frame = False

        self._show_mocap = True  # Show / hide mocap bodies.
        self._transparent = False  # Make everything transparent.

        # this variable is estamated as a running average.
        self._time_per_render = 1 / 60.0
        self._hide_overlay = False  # hide the entire overlay.
        self._user_overlay = {}

        # these variables are for changing the x,y,z location of an object
        # either 0 (no press), or +/-1 are returned, the scaling is up to the
        # user end
        self.target = np.zeros(3)
        self.old_target = np.zeros(3)
        self.scale = 0.05

        self.xyz_lowerbound = [-0.5, -0.5, 0.0]
        self.xyz_upperbound = [0.5, 0.5, 0.7]
        self.rlim = [0.4, 1.0]

        # let the user define what the robot should do, pick up object, drop it
        # off, or reach to a target
        self.reach_mode = 'reach_target'
        self.reach_mode_changed = False

        # sets if we're in demo mode
        self.toggle_demo = False
        self.key_pressed = False

        # visualization of position and orientation of path planner / filter
        self.path_vis = False

        # allow for printing to top right from user side
        self.custom_print = ""

        # toggle for adaptation
        self.adapt = False

        # manual toggle of gripper status
        self.gripper = 1

        # display mujoco default text
        self.display_all_text = display_all_text

        # scaling factor on external force to apply to body
        self.external_force = 0

        # additional mass for pick up object
        self.additional_mass = 0

        # various gravities
        self.gravities = {
            'earth': np.array([0, 0, -9.81]),
            'moon': np.array([0, 0, -1.62]),
            'mars': np.array([0, 0, -3.71]),
            'jupiter': np.array([0, 0, -24.92]),
            'ISS': np.array([0, 0, 0]),
            }
        self.planet = 'earth'

        # world gravity
        self.gravity = self.gravities['earth']

        self.restart_sim = False
        self.move_elbow = False
        self.target_moved = False
        self.elbow_force = np.zeros(6)

        self.display_hotkeys = False


    def render(self):
        """
        Render the current simulation state to the screen or off-screen buffer.
        Call this in your main loop.
        """

        def render_inner_loop(self):
            render_start = time.time()

            self._overlay.clear()
            if not self._hide_overlay:
                for k, v in self._user_overlay.items():
                    self._overlay[k] = copy.deepcopy(v)
                self._create_full_overlay()
            super().render()
            if self._record_video:
                frame = self._read_pixels_as_in_window()
                self._video_queue.put(frame)
            else:
                self._time_per_render = 0.9 * self._time_per_render + \
                    0.1 * (time.time() - render_start)

        self._user_overlay = copy.deepcopy(self._overlay)
        # Render the same frame if paused.
        if self._paused:
            while self._paused:
                render_inner_loop(self)
                if self._advance_by_one_step:
                    self._advance_by_one_step = False
                    break
        else:
            # inner_loop runs "_loop_count" times in expectation (where "_loop_count" is a float).
            # Therefore, frames are displayed in the real-time.
            self._loop_count += self.sim.model.opt.timestep * self.sim.nsubsteps / \
                (self._time_per_render * self._run_speed)
            if self._render_every_frame:
                self._loop_count = 1
            while self._loop_count > 0:
                render_inner_loop(self)
                self._loop_count -= 1
        # Markers and overlay are regenerated in every pass.
        self._markers[:] = []
        self._overlay.clear()

    def _read_pixels_as_in_window(self):
        # Reads pixels with markers and overlay from the same camera as screen.
        resolution = glfw.get_framebuffer_size(
            self.sim._render_context_window.window)

        resolution = np.array(resolution)
        resolution = resolution * min(1000 / np.min(resolution), 1)
        resolution = resolution.astype(np.int32)
        resolution -= resolution % 16
        if self.sim._render_context_offscreen is None:
            self.sim.render(resolution[0], resolution[1])
        offscreen_ctx = self.sim._render_context_offscreen
        window_ctx = self.sim._render_context_window
        # Save markers and overlay from offscreen.
        saved = [copy.deepcopy(offscreen_ctx._markers),
                 copy.deepcopy(offscreen_ctx._overlay),
                 rec_copy(offscreen_ctx.cam)]
        # Copy markers and overlay from window.
        offscreen_ctx._markers[:] = window_ctx._markers[:]
        offscreen_ctx._overlay.clear()
        offscreen_ctx._overlay.update(window_ctx._overlay)
        rec_assign(offscreen_ctx.cam, rec_copy(window_ctx.cam))

        img = self.sim.render(*resolution)
        img = img[::-1, :, :] # Rendered images are upside-down.
        # Restore markers and overlay to offscreen.
        offscreen_ctx._markers[:] = saved[0][:]
        offscreen_ctx._overlay.clear()
        offscreen_ctx._overlay.update(saved[1])
        rec_assign(offscreen_ctx.cam, saved[2])
        return img

    def _create_full_overlay(self):
        if self.display_all_text:
            if self._render_every_frame:
                self.add_overlay(const.GRID_TOPLEFT, "", "")
            else:
                self.add_overlay(const.GRID_TOPLEFT, "Run speed = %.3f x real time" %
                                self._run_speed, "[S]lower, [F]aster")
            if self._paused is not None:
                if not self._paused:
                    self.add_overlay(const.GRID_TOPLEFT, "Stop", "[Space]")
                else:
                    self.add_overlay(const.GRID_TOPLEFT, "Start", "[Space]")
            self.add_overlay(const.GRID_TOPLEFT, "[H]ide Menu", "")

            self.add_overlay(const.GRID_BOTTOMLEFT, "FPS", "%d%s" %
                            (1 / self._time_per_render, extra))
            self.add_overlay(const.GRID_BOTTOMLEFT, "Solver iterations", str(
                self.sim.data.solver_iter + 1))
            step = round(self.sim.data.time / self.sim.model.opt.timestep)
            self.add_overlay(const.GRID_BOTTOMRIGHT, "Step", str(step))
            self.add_overlay(const.GRID_BOTTOMRIGHT, "timestep", "%.5f" % self.sim.model.opt.timestep)
            self.add_overlay(const.GRID_BOTTOMRIGHT, "n_substeps", str(self.sim.nsubsteps))
            self.add_overlay(const.GRID_TOPLEFT, "Toggle geomgroup visibility", "0-4")

        # CUSTOM KEYS
        self.add_overlay(const.GRID_TOPLEFT, "Toggle adaptation", "[LEFT SHIFT]")
        self.add_overlay(const.GRID_TOPLEFT, "Move target along X/Y", "RIGHT/LEFT/UP/DOWN")
        self.add_overlay(const.GRID_TOPLEFT, "Move target along Z", "[ALT+ UP/DOWN]")
        self.add_overlay(const.GRID_TOPLEFT, "Follow target", "[F1]")
        self.add_overlay(const.GRID_TOPLEFT, "Pick up object", "[F2]")
        self.add_overlay(const.GRID_TOPLEFT, "Dumbbell mass +1kg", "[u]")
        self.add_overlay(const.GRID_TOPLEFT, "Dumbbell mass -1kg", "[y]")
        self.add_overlay(const.GRID_TOPLEFT, "Mars Gravity", "[1]")
        self.add_overlay(const.GRID_TOPLEFT, "Earth Gravity", "[2]")
        self.add_overlay(const.GRID_TOPLEFT, "Moon Gravity", "[3]")
        self.add_overlay(const.GRID_TOPLEFT, "Jupiter Gravity", "[4]")
        self.add_overlay(const.GRID_TOPLEFT, "ISS Gravity", "[5]")

        self.add_overlay(const.GRID_TOPRIGHT, "Adaptation: %s"%self.adapt, "")
        self.add_overlay(const.GRID_TOPRIGHT, "%s"%self.reach_mode, "")
        self.add_overlay(const.GRID_TOPRIGHT, "Elbow force ", ''.join('%.2f ' % val for val in self.elbow_force[:3]))
        self.add_overlay(const.GRID_TOPRIGHT, "%s"%self.custom_print, "")

    def key_callback(self, window, key, scancode, action, mods):
        dx, dy, dz = 0, 0, 0
        arrow_keys = False

        # on button press (for button holding)
        if action != glfw.RELEASE:
            self.key_pressed = True
            # adjust object location up / down
            # Z
            if glfw.get_key(window, glfw.KEY_LEFT_ALT):
                if key == glfw.KEY_UP:
                    dz = 1
                    arrow_keys = True
                elif key == glfw.KEY_DOWN:
                    dz = -1
                    arrow_keys = True
            else:
                # X
                if key == glfw.KEY_LEFT:
                    dx = -1
                    arrow_keys = True
                elif key == glfw.KEY_RIGHT:
                    dx = 1
                    arrow_keys = True
                # Y
                if key == glfw.KEY_UP:
                    dy = 1
                    arrow_keys = True
                elif key == glfw.KEY_DOWN:
                    dy = -1
                    arrow_keys = True

            super().key_callback(window, key, scancode, action, mods)

        # on button release (click)
        else:
            self.key_pressed = True

            if key == glfw.KEY_H:  # hides all overlay.
                self._hide_overlay = not self._hide_overlay

            # elif key == glfw.KEY_SPACE and self._paused is not None:  # stops simulation.
            #     self._paused = not self._paused
            elif key == glfw.KEY_SPACE:
                self.display_hotkeys = not self.display_hotkeys

            # adjust object location up / down
            # Z
            elif glfw.get_key(window, glfw.KEY_LEFT_ALT) and key == glfw.KEY_UP:
                dz = 1
                arrow_keys = True
            elif glfw.get_key(window, glfw.KEY_LEFT_ALT) and key == glfw.KEY_DOWN:
                dz = -1
                arrow_keys = True
            # X
            elif key == glfw.KEY_LEFT:
                dx = -1
                arrow_keys = True
            elif key == glfw.KEY_RIGHT:
                dx = 1
                arrow_keys = True
            # Y
            elif key == glfw.KEY_UP:
                dy = 1
                arrow_keys = True
            elif key == glfw.KEY_DOWN:
                dy = -1
                arrow_keys = True

            # user command to reach to target
            elif key == glfw.KEY_F1:
                self.reach_mode = 'reach_target'
                self.reach_mode_changed = True
            # user command to pick up object
            elif key == glfw.KEY_F2:
                self.reach_mode = 'pick_up'
                self.reach_mode_changed = True
            # user command to drop off object
            elif key == glfw.KEY_F3:
                self.reach_mode = 'drop_off'
                self.reach_mode_changed = True
            elif key == glfw.KEY_F4:
                self.path_vis = not self.path_vis

            # toggle adaptation
            elif key == glfw.KEY_LEFT_SHIFT:
                self.adapt = not self.adapt

            # scaling factor on external force
            elif key == glfw.KEY_G:
                self.external_force += 1

            elif key == glfw.KEY_B:
                self.external_force -= 1

            # scaling factor on external force
            elif key == glfw.KEY_U:
                self.additional_mass = 1

            elif key == glfw.KEY_Y:
                self.additional_mass = -1

            # set the world gravity
            elif key == glfw.KEY_1:
                self.planet = 'mars'

            elif key == glfw.KEY_2:
                self.planet = 'earth'

            elif key == glfw.KEY_3:
                self.planet = 'moon'

            elif key == glfw.KEY_4:
                self.planet = 'jupiter'

            elif key == glfw.KEY_5:
                self.planet = 'ISS'

            elif key == glfw.KEY_F5:
                self.restart_sim = True

            elif key == glfw.KEY_ENTER:
                self.toggle_demo = True

            elif key == glfw.KEY_TAB:
                self.move_elbow = not self.move_elbow

            super().key_callback(window, key, scancode, action, mods)

        if arrow_keys:
            if self.move_elbow:
                self.elbow_force[:3] += np.array([dx, dy, dz]) * 10
            else:
                self.target += self.scale * np.array([dx, dy, dz])

                # check that we're within radius thresholds, if set
                if self.rlim[0] is not None:
                    if np.linalg.norm(self.target) < self.rlim[0]:
                        self.target = self.old_target

                if self.rlim[1] is not None:
                    if np.linalg.norm(self.target) > self.rlim[1]:
                        self.target= self.old_target

                self.target = np.clip(
                    self.target, self.xyz_lowerbound, self.xyz_upperbound)

                self.target_moved = True
