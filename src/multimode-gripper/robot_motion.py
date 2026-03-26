from pyAgxArm import create_agx_arm_config, AgxArmFactory
import warnings
import time
warnings.filterwarnings("ignore", category=DeprecationWarning) #Due to Chinese text in docstring of pyAgxArm
cfg = create_agx_arm_config(robot="piper", comm="can", channel="can0")
print(cfg)
robot = AgxArmFactory.create_arm(cfg)
end_effector = robot.init_effector(robot.OPTIONS.EFFECTOR.AGX_GRIPPER)
robot.connect()
time.sleep(0.5)
print("robotic arm is_ok =", robot.is_ok())
print("effector is_ok =", end_effector.is_ok())
robot.set_tcp_offset([0.0, 0.0, 0.12, 0.0, 0.0, 0.0])


tcp = robot.get_tcp_pose()


gs = end_effector.get_gripper_status()
if gs is not None:
    print("width(m)=", gs.msg.width, "force(N)=", gs.msg.force)

initial_flange_pose = robot.get_tcp2flange_pose(tcp.msg)
print(initial_flange_pose)
new_flange_pose = [0.0617, -0.000796, 0.254825, -2.229, -1.19993, -0.96075]
newer_flange_pose = [0.170657, 0.001744, 0.20840499, -3.0697549, -0.0625526, -0.36459]
print(new_flange_pose)

robot.enable()
robot.set_speed_percent(10)


robot.set_payload(robot.OPTIONS.PAYLOAD.FULL)

def move_and_wait(robot, new_flange_pose):
    robot.move_p(new_flange_pose)
    start_t = time.monotonic()
    time.sleep(0.1)
    while True:
        status = robot.get_arm_status()
        if status is not None and status.msg.motion_status == 0:
            print("done")
            break
        if time.monotonic() - start_t > 20.0:
            print("timeout（20s）")
            break
        time.sleep(0.1)
#move_and_wait(robot, new_flange_pose)
#move_and_wait(robot, newer_flange_pose)

end_effector.move_gripper(width=0.06, force=1.0)
time.sleep(1.0)

end_effector.move_gripper(width=0.02, force=1.0)