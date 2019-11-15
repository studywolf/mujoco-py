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
from mujoco_py.xbox_controller import XboxController

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

        self.xyz_lowerbound = [-0.5, -0.5, 0.001]
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
        # self.additional_mass = 0
        self.dumbbell_mass_index = 0

        # various gravities
        self.gravities = {
            'earth': np.array([0, 0, -9.81]),
            'moon': np.array([0, 0, -1.62]),
            'mars': np.array([0, 0, -3.71]),
            'jupiter': np.array([0, 0, -24.92]),
            'ISS': np.array([0, 0, 0]),
            }
        # ids for cycling through planets with arrow keys
        self.planet_names = {
            0: 'mars', 1: 'earth', 2: 'moon', 3: 'jupiter', 4: 'ISS'}
        self.planet_ids = {
            'mars': 0, 'earth': 1, 'moon': 2, 'jupiter': 3, 'ISS': 4}

        self.planet = 'earth'
        self.planet_id = self.planet_ids[self.planet]

        # world gravity
        self.gravity = self.gravities['earth']

        self.restart_sim = False
        self.move_elbow = False
        self.target_moved = False
        self.elbow_force = np.zeros(6)

        self.display_hotkeys = False


        # xbox controller params
        self.use_controller = False
        # mapping between keyboard keys and xbox controller
        self.kb_xbox_mapping = {}
        self.xboxContId = None
        self.xbox_val = None
        # scaling for target step movement, when using jostick this value will drop
        self.joystick_scale = 1
        # with the xbox controller, use R and L triggers to switch to controlling
        # target and elbow (respectively) z direction with joystick
        # hitting trigger toggles between controlling z and xy
        self.target_z_toggle = False
        self.elbox_z_toggle = False

        # mapping glfw to command
        self.keys = {
            'mars': glfw.KEY_1,
            'earth': glfw.KEY_2,
            'moon': glfw.KEY_3,
            'jupiter': glfw.KEY_4,
            'ISS': glfw.KEY_5,

            'left': glfw.KEY_LEFT,
            'right': glfw.KEY_RIGHT,
            'forward': glfw.KEY_UP,
            'backward': glfw.KEY_DOWN,
            'z_toggle': glfw.KEY_LEFT_ALT,

            'hot_keys': glfw.KEY_SPACE,
            'exit': glfw.KEY_ESCAPE,
            'hide_overlay': glfw.KEY_H,
            'restart': glfw.KEY_F5,
            'demo_toggle': glfw.KEY_ENTER,

            'reach_target': glfw.KEY_F1,
            'pick_up': glfw.KEY_F2,
            'drop_off': glfw.KEY_F3,

            'path_vis': glfw.KEY_F4,

            'adapt': glfw.KEY_LEFT_SHIFT,
            'move_elbow': glfw.KEY_TAB,
            'mass_up': glfw.KEY_U,
            'mass_down': glfw.KEY_Y,
            }

        # mapping xbox id to key
        self.id2xbox = {
            0: 'lthumbx',
            1: 'lthumby',
            2: 'ltrigger',
            3: 'rthumbx',
            4: 'rthumby',
            5: 'rtrigger',
            6: 'A',
            7: 'B',
            8: 'X',
            9: 'Y',
            10: 'LB',
            11: 'RB',
            12: 'back',
            13: 'start',
            14: 'xbox',
            15: 'lthumb',
            16: 'rthumb',
            17: 'dpad'
            }

        # mapping xbox key to action
        self.buttons_to_glfw = {
            # 'lthumbx',
            # 'lthumby',
            # 'rthumbx',
            # 'rthumby',
            # 'rtrigger': glfw.KEY_,
            # 'ltrigger': glfw.KEY_,
            'A': None,
            'B': self.keys['restart'],
            'X': self.keys['move_elbow'],
            'Y': self.keys['demo_toggle'],
            'LB': self.keys['pick_up'],
            'RB': self.keys['reach_target'],
            'back': self.keys['exit'],
            'start': self.keys['hot_keys'],
            'xbox': self.keys['adapt'],
            # 'lthumb': glfw.KEY_,
            # 'rthumb': glfw.KEY_,
            # 'dpad'
            }


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
            # adjust object location up / down
            # Z
            if glfw.get_key(window, self.keys['z_toggle']) or self.target_z_toggle:
                if key == self.keys['forward']:
                    dz = 1 * self.joystick_scale
                    arrow_keys = True
                    self.key_pressed = True
                elif key == self.keys['backward']:
                    dz = -1 * self.joystick_scale
                    arrow_keys = True
                    self.key_pressed = True
            else:
                # X
                if key == self.keys['left']:
                    dx = -1 * self.joystick_scale
                    arrow_keys = True
                    self.key_pressed = True
                elif key == self.keys['right']:
                    dx = 1 * self.joystick_scale
                    arrow_keys = True
                    self.key_pressed = True
                # Y
                if key == self.keys['forward']:
                    dy = 1 * self.joystick_scale
                    arrow_keys = True
                    self.key_pressed = True
                elif key == self.keys['backward']:
                    dy = -1 * self.joystick_scale
                    arrow_keys = True
                    self.key_pressed = True

            super().key_callback(window, key, scancode, action, mods)

        # on button release (click)
        elif action == glfw.RELEASE:

            if key == self.keys['hide_overlay']:  # hides all overlay.
                self._hide_overlay = not self._hide_overlay
            elif key == self.keys['hot_keys']: # and self._paused is not None:  # stops simulation.
                # self._paused = not self._paused
                self.display_hotkeys = not self.display_hotkeys

            # adjust object location up / down
            # Z
            elif ((glfw.get_key(window, self.keys['z_toggle']) )#or self.target.z_toggle)
                  and key == self.keys['forward']):
                dz = 1 * self.joystick_scale
                arrow_keys = True
                self.key_pressed = True
            elif ((glfw.get_key(window, self.keys['z_toggle']) )#or self.target_z_toggle)
                  and key == self.keys['backward']):
                dz = -1 * self.joystick_scale
                arrow_keys = True
                self.key_pressed = True
            # X
            elif key == self.keys['left']:
                dx = -1 * self.joystick_scale
                arrow_keys = True
                self.key_pressed = True
            elif key == self.keys['right']:
                dx = 1 * self.joystick_scale
                arrow_keys = True
                self.key_pressed = True
            # Y
            elif key == self.keys['forward']:
                dy = 1 * self.joystick_scale
                arrow_keys = True
                self.key_pressed = True
            elif key == self.keys['backward']:
                dy = -1 * self.joystick_scale
                arrow_keys = True
                self.key_pressed = True

            # user command to reach to target
            elif key == self.keys['reach_target']:
                self.reach_mode = 'reach_target'
                self.reach_mode_changed = True
                self.key_pressed = True
            # user command to pick up object
            elif key == self.keys['pick_up']:
                self.reach_mode = 'pick_up'
                self.reach_mode_changed = True
                self.key_pressed = True
            # user command to drop off object
            elif key == self.keys['drop_off']:
                self.reach_mode = 'drop_off'
                self.reach_mode_changed = True
                self.key_pressed = True
            # elif key == self.keys['path_vis']:
            #     self.path_vis = not self.path_vis

            # toggle adaptation
            elif key == self.keys['adapt']:
                self.adapt = not self.adapt
                self.key_pressed = True

            # # scaling factor on external force
            # elif key == self.keys['force_up']:
            #     self.external_force += 1
            #
            # elif key == self.keys['force_down']:
            #     self.external_force -= 1

            # scaling factor on external force
            elif key == self.keys['mass_up']:
                print('so masseus')
                # self.additional_mass = 1
                self.dumbbell_mass_index += 1

            elif key == self.keys['mass_down']:
                print('no masseus')
                # self.additional_mass = -1
                self.dumbbell_mass_index -= 1

            # set the world gravity
            elif key == self.keys['mars']:
                self.planet = 'mars'

            elif key == self.keys['earth']:
                self.planet = 'earth'

            elif key == self.keys['moon']:
                self.planet = 'moon'

            elif key == self.keys['jupiter']:
                self.planet = 'jupiter'

            elif key == self.keys['ISS']:
                self.planet = 'ISS'

            elif key == self.keys['restart']:
                self.restart_sim = True

            elif key == self.keys['demo_toggle']:
                self.toggle_demo = True
                self.key_pressed = True

            elif key == self.keys['move_elbow']:
                self.move_elbow = not self.move_elbow

            print('called back!!')

            super().key_callback(window, key, scancode, action, mods)

        if arrow_keys:
            if self.move_elbow:
                self.elbow_force[:3] += np.array([dx, dy, dz]) * 10
            else:
                self.old_target = np.copy(self.target)
                self.target += self.scale * np.array([dx, dy, dz])

                # check that we're within radius thresholds, if set
                if self.rlim[0] is not None:
                    if np.linalg.norm(self.target) < self.rlim[0]:
                        self.target = self.old_target

                if self.rlim[1] is not None:
                    if np.linalg.norm(self.target) > self.rlim[1]:
                        self.target = self.old_target

                self.target = np.clip(
                    self.target, self.xyz_lowerbound, self.xyz_upperbound)

                self.target_moved = True

    def xbox_vals(self, xboxControlId=None, value=None):
        self.xboxContId = xboxControlId
        self.xbox_val = value


    def setup_xbox_controller(self, deadzone=75, scale=100):
        self.deadzone = deadzone
        #setup xbox controller, set out the deadzone and scale, also invert the Y Axis (for some reason in Pygame negative is up - wierd!
        self.xboxCont = XboxController(
            self.xbox_vals, deadzone=self.deadzone, scale=scale, invertYAxis=True)
        self.use_controller = True
        self.joystick_scale = 0.1


    def xbox_conversion(self):

        key = None
        action = None

        button = self.id2xbox[self.xboxContId]
        print('\n', button)
        print(self.xbox_val)
        # thumbsticks
        if 'rthumb' in button:
            # top right quadrant
            if sum(abs(self.xbox_val)) > 10:
                if self.xbox_val[0] >= 0 and self.xbox_val[1] >= 0:
                    if abs(self.xbox_val[0]) > abs(self.xbox_val[1]):
                        key = self.keys['right']
                    else:
                        key = self.keys['forward']

                # top left quadrant
                elif self.xbox_val[0] < 0 and self.xbox_val[1] >= 0:
                    if abs(self.xbox_val[0]) > abs(self.xbox_val[1]):
                        key = self.keys['left']
                    else:
                        key = self.keys['forward']

                # bottom left quadrant
                elif self.xbox_val[0] < 0 and self.xbox_val[1] < 0:
                    if abs(self.xbox_val[0]) > abs(self.xbox_val[1]):
                        key = self.keys['left']
                    else:
                        key = self.keys['backward']

                # bottom right quadrant
                elif self.xbox_val[0] >= 0 and self.xbox_val[1] < 0:
                    if abs(self.xbox_val[0]) > abs(self.xbox_val[1]):
                        key = self.keys['right']
                    else:
                        key = self.keys['backward']

                action = 1

        #TODO: this needs to be mapped to other keys so we can control the elbow
        # you'll need to add the commands to self.keys and reference them here,
        # you'll also need to add the commands to the callback above
        # the a and x keys are still available, it might be easiest to use x to
        # toggle move elbow

        elif 'lthumb' in button:
            pass
        # if 'lthumb' in button:
        #     # top right quadrant
        #     if self.xbox_val[0] >= 0 and self.xbox_val[1] >= 0:
        #         if abs(self.xbox_val[0]) > abs(self.xbox_val[1]):
        #             key = self.keys['right']
        #         else:
        #             key = self.keys['forward']
        #
        #     # top left quadrant
        #     elif self.xbox_val[0] < 0 and self.xbox_val[1] >= 0:
        #         if abs(self.xbox_val[0]) > abs(self.xbox_val[1]):
        #             key = self.keys['left']
        #         else:
        #             key = self.keys['forward']
        #
        #     # bottom left quadrant
        #     elif self.xbox_val[0] < 0 and self.xbox_val[1] < 0:
        #         if abs(self.xbox_val[0]) > abs(self.xbox_val[1]):
        #             key = self.keys['left']
        #         else:
        #             key = self.keys['backward']
        #
        #     # bottom right quadrant
        #     elif self.xbox_val[0] >= 0 and self.xbox_val[1] < 0:
        #         if abs(self.xbox_val[0]) > abs(self.xbox_val[1]):
        #             key = self.keys['right']
        #         else:
        #             key = self.keys['backward']
        #
        #     action = 1

        # triggers
        elif button == 'rtrigger':
            if self.xbox_val > 95:
                self.target_z_toggle = not self.target_z_toggle
                key = None
                action = None

        elif button == 'ltrigger':
            key=None
            action=None

        # dpad
        elif button == 'dpad':
            if self.xbox_val[1] == 1:
                print('mass up')
                key = self.keys['mass_up']
                action = glfw.RELEASE
            elif self.xbox_val[1] == -1:
                key = self.keys['mass_down']
                action = glfw.RELEASE
            elif self.xbox_val[0] == -1:
                key = self.keys[
                    self.planet_names[max(0, (self.planet_ids[self.planet]-1))]]
                action = glfw.RELEASE
            elif self.xbox_val[0] == 1:
                key = self.keys[
                    self.planet_names[min(4, (self.planet_ids[self.planet]+1))]]
                action = glfw.RELEASE

        # buttons
        else:
            key = self.buttons_to_glfw[button]
            action = self.xbox_val

        return key, action


    def xbox_callback(self):
        # check for keypress
        if self.use_controller:
            self.xboxCont._start()
            if self.xboxContId is not None:
                key, action = self.xbox_conversion()
                print('received: ', key)
                if key is not None and action is not None:
                    print('running callback')
                    self.key_callback(window=self.window, key=key, scancode=None, action=action, mods=None)
            # only reset when not thumbstick
            # if (self.xboxContId != 0
            #         or self.xboxContId != 1
            #         or self.xboxContId != 3
            #         or self.xboxContId != 4):
            self.xboxContId = None
            self.xbox_val = None

