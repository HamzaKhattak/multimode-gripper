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
        self.gripper_path = []
        self.time_track = time.perf_counter()
        self.minmove_correction = 0


    def move_and_wait(self, new_flange_pose):
        self.robot.move_l(new_flange_pose)
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
            self.robot.move_l(new_flange_pose)
            self._last_commanded_pose = new_flange_pose

        status = self.robot.get_arm_status()
        if status is not None and status.msg.motion_status == 0:
            return True
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
    
    def move_gripper_slowly(self, target_position: float, force : float,force_threshold: float, speed: float, min_move: float)-> (bool, float, float):
        '''
        Move the gripper to the target position in small steps to ensure a smooth motion.
        Should be called in a loop until it returns True, which indicates the target position or force has been reached.
        Returns true when the target position is reached, False otherwise. Also returns the position and force for logging purposes.
        '''
        #If this is the first call to the function, need to reset the timer and move correction
        if self._last_commanded_gripper_pos == target_position:
            self.time_track = time.perf_counter() #Reset the timer if it is the first time calling a move to this target position
            self._last_commanded_gripper_pos = target_position
            self.distance_to_move_correction = 0


        current_position, current_force = self.get_gripper_wf()
        distance_to_target = abs(target_position - current_position)
        #If we are already at the target position or the force is greater than or equal to the target force, we are done and can return true.
        if distance_to_target == 0 or current_force >= force_threshold:
            return True, current_position, current_force
        
        #Find how much to move based on speed and time from last call, and apply correction from previous moves if we were below the minimum move threshold    
        current_time = time.perf_counter()
        dt = current_time - self.time_track
        self.time_track = current_time
        distance_to_move_uncorrected = speed * dt

        #If the distance to move is greater than the distance to the target, we can just move to the target and be done.
        if distance_to_move_uncorrected <= distance_to_target:
            distance_to_move = distance_to_target
            self.distance_to_move_correction = 0

        #If the distance to move is larger than the minimum move threshold, we can move the full delta.
        elif distance_to_move_uncorrected > min_move:
            distance_to_move = distance_to_move_uncorrected
            self.distance_to_move_correction = 0

        #If the distance to move is smaller, we keep track and add it to the next move until it exceeds the minimum move threshold. This is to avoid very small movements.
        elif distance_to_move_uncorrected < min_move:
            self.distance_to_move_correction += self.distance_to_move_correction
            distance_to_move = 0
        
        #Find where to move
        if current_position < target_position:
            new_position = current_position + distance_to_move
        else:
            new_position = current_position - distance_to_move
        self.end_effector.move_gripper(new_position, force)
        return False, current_position, current_force




