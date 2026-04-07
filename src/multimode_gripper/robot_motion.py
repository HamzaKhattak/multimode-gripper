from turtle import delay

from pyAgxArm import create_agx_arm_config, AgxArmFactory
import warnings
import time
warnings.filterwarnings("ignore", category=DeprecationWarning) #Due to Chinese text in docstring of pyAgxArm
import numpy as np
class RobotMotion:
    def __init__(self,end_effector_use=False,speed_percent=10,payload="full"):
        cfg = create_agx_arm_config(robot="piper", comm="can", channel="can0")
        self.robot = AgxArmFactory.create_arm(cfg)
        if end_effector_use:
            self.end_effector = self.robot.init_effector(self.robot.OPTIONS.EFFECTOR.AGX_GRIPPER)
        self.robot.connect()
        time.sleep(0.5)
        print("robotic arm is_ok =", self.robot.is_ok())

        if end_effector_use:
            print("effector is_ok =", self.end_effector.is_ok())
            self.gripper_pos, self.gripper_force = self.get_gripper_wf()
        
        self.robot.enable()
        self.robot.set_speed_percent(speed_percent)
        if payload == "full":
            self.robot.set_payload(self.robot.OPTIONS.PAYLOAD.FULL)
        self._last_commanded_pose = None
        self._last_commanded_gripper_pos = None
        self._last_slow_gripper_setpoint = None
        self._slow_gripper_force_baseline = None
        self._force_contact_detected = False
        self._last_release_target_pos = None
        self._last_release_setpoint = None
        self._release_time_track = time.perf_counter()
        self._release_distance_correction = 0
        self.gripper_path = []
        self.time_track = time.perf_counter()
        self.distance_to_move_correction = 0


    def move_and_wait(self, new_flange_pose):
        self.robot.move_p(new_flange_pose)
        start_t = time.monotonic()
        time.sleep(0.1)
        while True:
            status = self.robot.get_arm_status()
            if status is not None and status.msg.motion_status == 0:
                print("done")
                break
            if time.monotonic() - start_t > 20.0:
                print("timeout（20s）")
                break
            time.sleep(0.1)

    def move_non_blocking(self, new_flange_pose) -> bool:
        """Send a move command only when the target pose changes.

        Returns True when the robot has finished moving, False while still in motion.
        """
        if new_flange_pose != self._last_commanded_pose:
            print(f"Sending move command to pose: {new_flange_pose}")
            self.robot.move_p(new_flange_pose)
            self._last_commanded_pose = new_flange_pose
            self._movement_in_progress = True  # Mark that we sent a move command
            return False  # Movement just started, not complete yet

        status = self.robot.get_arm_status()
        if status is not None:
            motion_status = status.msg.motion_status
            # Debug: print status occasionally
            if hasattr(self, '_debug_count'):
                self._debug_count += 1
            else:
                self._debug_count = 0
            
            if self._debug_count % 500 == 0:  # Print every 500 calls
                print(f"Robot motion status: {motion_status}")
            
            # Wait for movement to complete (motion_status == 0 AND we had movement in progress)
            if motion_status == 0 and getattr(self, '_movement_in_progress', False):
                print("Movement completed!")
                self._movement_in_progress = False
                return True
        else:
            print("Could not get robot status")
        return False

    def get_gripper_wf(self):
        '''
        Returns the current width and force of the gripper.
        '''
        if not hasattr(self, 'end_effector'):
             raise Exception("End effector not initialized.")
        gs = self.end_effector.get_gripper_status()
        return gs.msg.width , gs.msg.force 
             
    def set_gripper_position_simple(self, new_position,force):
        '''
        Set the gripper position to new_position with the given force. Only sends a command if the position changes.
        '''
        self.end_effector.move_gripper(new_position, force)
    
    def open_gripper(self):
        '''
        Open the gripper fully with the given force.
        '''
        open_position = .095
        self.end_effector.move_gripper(open_position, 3)

    def _get_max_release_force(self, measured_force: float) -> float:
        '''
        Return fixed max force command for release motions.
        '''
        return 3.0

    def release_slowly(
        self,
        release_force_threshold: float,
        slow_speed: float,
        fast_speed: float,
        min_move: float,
        target_position: float = 0.095,
    ) -> tuple[bool, float, float]:
        '''
        Non-blocking gripper release.

        Open the gripper toward target_position. While gripping force is above
        release_force_threshold, move with slow_speed. Once force is below threshold,
        move with fast_speed. Call repeatedly until the returned `done` is True.
        '''
        current_position, current_force = self.get_gripper_wf()

        if self._last_release_target_pos != target_position:
            self._last_release_target_pos = target_position
            self._last_release_setpoint = current_position
            self._release_time_track = time.perf_counter()
            self._release_distance_correction = 0

        # Use the commanded trajectory as reference so progress continues even
        # when measured width updates lag while force is still decaying.
        direction = 1.0 if target_position >= current_position else -1.0
        if self._last_release_setpoint is None:
            self._last_release_setpoint = current_position

        if direction > 0:
            base_position = max(current_position, self._last_release_setpoint)
        else:
            base_position = min(current_position, self._last_release_setpoint)

        distance_to_target = abs(target_position - base_position)
        if distance_to_target <= 0:
            return True, current_position, current_force

        active_speed = fast_speed if current_force <= release_force_threshold else slow_speed

        current_time = time.perf_counter()
        dt = current_time - self._release_time_track
        self._release_time_track = current_time
        proposed_move = (active_speed * dt) + self._release_distance_correction

        if proposed_move >= distance_to_target:
            distance_to_move = distance_to_target
            self._release_distance_correction = 0
        elif proposed_move >= min_move:
            distance_to_move = proposed_move
            self._release_distance_correction = 0
        else:
            self._release_distance_correction = proposed_move
            distance_to_move = 0

        if distance_to_move == 0:
            return False, current_position, current_force

        if direction > 0:
            new_position = min(target_position, base_position + distance_to_move)
        else:
            new_position = max(target_position, base_position - distance_to_move)

        self._last_release_setpoint = new_position
        self.end_effector.move_gripper(new_position, 3.0)
        return False, current_position, current_force
    
    def grab_slowly(self, target_position: float, force : float,force_threshold: float, speed: float, min_move: float)-> tuple[bool, float, float]:
        '''
        Move the gripper to the target position in small steps to ensure a smooth motion.
        Should be called in a loop until it returns True, which indicates the target position or force has been reached.
        Returns true when the target position is reached, False otherwise. Also returns the position and force for logging purposes.
        '''
        current_position, current_force = self.get_gripper_wf()
        # If this is the first call for this target, reset timing and correction state.
        if self._last_commanded_gripper_pos != target_position:
            self.time_track = time.perf_counter()
            self._last_commanded_gripper_pos = target_position
            self.distance_to_move_correction = 0
            self._last_slow_gripper_setpoint = current_position
            self._slow_gripper_force_baseline = current_force
            self._force_contact_detected = False

        # Detect first contact and switch to cumulative setpoint updates so we do not
        # keep sending the same command when measured position stalls under load.
        if self._slow_gripper_force_baseline is None:
            self._slow_gripper_force_baseline = current_force

        if np.isfinite(force_threshold):
            contact_force_delta_threshold = max(0.05, 0.1 * max(abs(force_threshold), 1.0))
        else:
            contact_force_delta_threshold = max(0.05, 0.1 * max(abs(force), 1.0))

        force_delta_from_start = abs(current_force - self._slow_gripper_force_baseline)
        if force_delta_from_start >= contact_force_delta_threshold:
            self._force_contact_detected = True

        direction = 1.0 if target_position >= current_position else -1.0
        if self._last_slow_gripper_setpoint is None:
            self._last_slow_gripper_setpoint = current_position

        if self._force_contact_detected:
            if direction > 0:
                base_position = max(current_position, self._last_slow_gripper_setpoint)
            else:
                base_position = min(current_position, self._last_slow_gripper_setpoint)
        else:
            base_position = current_position

        distance_to_target = abs(target_position - base_position)
        # If we are already at target (based on commanded trajectory) or at force threshold,
        # finish this motion.
        if distance_to_target <= 0 or current_force >= force_threshold:
            return True, current_position, current_force
        
        #Find how much to move based on speed and time from last call, and apply correction from previous moves if we were below the minimum move threshold    
        current_time = time.perf_counter()
        dt = current_time - self.time_track
        self.time_track = current_time
        distance_to_move_uncorrected = speed * dt
        proposed_move = distance_to_move_uncorrected + self.distance_to_move_correction

        # Snap to target if the proposed step would overshoot.
        if proposed_move >= distance_to_target:
            distance_to_move = distance_to_target
            self.distance_to_move_correction = 0

        # Otherwise move once we exceed the controller's minimum commandable move.
        elif proposed_move >= min_move:
            distance_to_move = proposed_move
            self.distance_to_move_correction = 0

        # Accumulate sub-threshold increments for a later command.
        else:
            self.distance_to_move_correction = proposed_move
            distance_to_move = 0

        if distance_to_move == 0:
            return False, current_position, current_force
        
        #Find where to move
        if direction > 0:
            new_position = base_position + distance_to_move
            if new_position > target_position:
                new_position = target_position
        else:
            new_position = base_position - distance_to_move
            if new_position < target_position:
                new_position = target_position

        self._last_slow_gripper_setpoint = new_position
        self.end_effector.move_gripper(new_position, force)
        return False, current_position, current_force




