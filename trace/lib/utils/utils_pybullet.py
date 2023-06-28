import pybullet as p
import time	

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

import pybullet as p
import pybullet_data
import json

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setPhysicsEngineParameter(numSolverIterations=200)

def randrange(n, vmin, vmax):
    '''
    Helper function to make an array of random numbers having shape (n, )
    with each number distributed Uniform(vmin, vmax).
    '''
    return (vmax - vmin)*np.random.rand(n) + vmin

def display_joints_matplotlib(joints_3d_positions):
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')

	# For each set of style and range settings, plot n random points in the box
	# defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
	c, m = ('r', 'o')
	
	xs = joints_3d_positions[:, 0]
	ys = joints_3d_positions[:, 1]
	zs = joints_3d_positions[:, 2]

	ax.scatter(xs, ys, zs, c=c, marker=m)


	for ind in range(joints_3d_positions.shape[0]):
		ax.text(xs[ind], ys[ind], zs[ind], str(ind), color=c)


	ax.set_xlabel('X Label')
	ax.set_ylabel('Y Label')
	ax.set_zlabel('Z Label')

	ax.set_xlim(-1, 1)
	ax.set_ylim(-1, 1)
	ax.set_zlim(-1, 1)

	plt.show()

def display_joints(joints_3d_positions):
	"""
	Displays a 3d scene, with a small sphere at each input 3d location.

	param: 3d_joints_positions, Nx3 numpy array of floats
	"""
	useMaximalCoordinates = 0
	p.connect(p.GUI)
	#p.loadSDF("stadium.sdf",useMaximalCoordinates=useMaximalCoordinates)
	# monastryId = concaveEnv =p.createCollisionShape(p.GEOM_MESH,fileName="samurai_monastry.obj",flags=p.GEOM_FORCE_CONCAVE_TRIMESH)
	# orn = p.getQuaternionFromEuler([1.5707963,0,0])
	# p.createMultiBody (0,monastryId, baseOrientation=orn)

	sphereRadius = 0.02
	colSphereId = p.createCollisionShape(p.GEOM_SPHERE,radius=sphereRadius)
	# colBoxId = p.createCollisionShape(p.GEOM_BOX,halfExtents=[sphereRadius,sphereRadius,sphereRadius])

	mass = 1
	visualShapeId = -1

	# joint_3d_positions = [[-0.00208288, -0.24095365,  0.02312295]]

	for ind_shpere, sphere_pos in enumerate(joints_3d_positions):
		print(ind_shpere, sphere_pos)

		sphereUid = p.createMultiBody(mass, colSphereId, visualShapeId, sphere_pos, useMaximalCoordinates=useMaximalCoordinates)
		p.changeDynamics(sphereUid,-1,spinningFriction=0.001, rollingFriction=0.001,linearDamping=0.0)
	
	p.setGravity(0,0,0)
	p.setRealTimeSimulation(1)

	while (1):
		# keys = p.getKeyboardEvents()
		#print(keys)
		time.sleep(0.01)

	return


def displayBulletFrames(motionFrames, actualSMPLPoses = None):
	"""
	Displays a scene and robot performing the movements described in motionFrames.
	Can be the "Frames" entry of a DeepMimic motion reference file.

	:param motionFrames N*43 array
	"""

	# p.connect(p.GUI)
	# p.setAdditionalSearchPath(pybullet_data.getDataPath())
	# p.setPhysicsEngineParameter(numSolverIterations=200)

	p.resetSimulation()

	p.loadURDF("plane.urdf",[0,0,-2])
	humanoid = p.loadURDF("humanoid/humanoid.urdf", globalScaling=0.25)

	numFrames = len(motionFrames)
	print("---> #frames = ", numFrames)

	# jointTypes = ["JOINT_REVOLUTE","JOINT_PRISMATIC",
	# 							"JOINT_SPHERICAL","JOINT_PLANAR","JOINT_FIXED"]								

	for j in range (p.getNumJoints(humanoid)):
		ji = p.getJointInfo(humanoid,j)
		targetPosition=[0]
		if (ji[2] == p.JOINT_SPHERICAL):
			targetPosition=[0,0,0,1]
		p.setJointMotorControlMultiDof(humanoid,j,p.POSITION_CONTROL,targetPosition, force=0)
		

	for j in range (p.getNumJoints(humanoid)):
		p.changeDynamics(humanoid,j,linearDamping=0, angularDamping=0)
		p.changeVisualShape(humanoid,j,rgbaColor=[1,1,1,1])


	p.changeVisualShape(humanoid,2,rgbaColor=[1,0,0,1])
	chest=1
	neck=2
	rightShoulder=3
	rightElbow=4
	leftShoulder=6
	leftElbow = 7
	rightHip = 9
	rightKnee=10
	rightAnkle=11
	leftHip = 12
	leftKnee=13
	leftAnkle=14

	p.getCameraImage(320,200)
	maxForce=1000

	frameReal = 0
	frameDuration = 0.001
	while (p.isConnected() and frameReal < numFrames-1):
		frameReal += frameDuration
		frame = int(frameReal)
		frameNext = frame+1
		if (frameNext >=  numFrames):
			frameNext = frame
		
		frameFraction = frameReal - frame
		# print("frameFraction=",frameFraction)
		# print("frame=",frame)
		# print("frameNext=", frameNext)
		
		#getQuaternionSlerp
		
		frameData = motionFrames[frame]
		frameDataNext = motionFrames[frameNext]
		
		#print("duration=",frameData[0])
		#print(pos=[frameData])
		
		basePos1Start = [frameData[1],frameData[2],frameData[3]]
		basePos1End = [frameDataNext[1],frameDataNext[2],frameDataNext[3]]
		basePos1 = [basePos1Start[0]+frameFraction*(basePos1End[0]-basePos1Start[0]), 
			basePos1Start[1]+frameFraction*(basePos1End[1]-basePos1Start[1]), 
			basePos1Start[2]+frameFraction*(basePos1End[2]-basePos1Start[2])]
		baseOrn1Start = [frameData[5],frameData[6], frameData[7],frameData[4]]
		baseOrn1Next = [frameDataNext[5],frameDataNext[6], frameDataNext[7],frameDataNext[4]]
		baseOrn1 = p.getQuaternionSlerp(baseOrn1Start,baseOrn1Next,frameFraction)
		#pre-rotate to make z-up
		y2zPos=[0,0,0.0]
		y2zOrn = p.getQuaternionFromEuler([1.57,0,0])
		basePos,baseOrn = p.multiplyTransforms(y2zPos, y2zOrn,basePos1,baseOrn1)

		p.resetBasePositionAndOrientation(humanoid, basePos,baseOrn)
		#	once=False
		chestRotStart = [frameData[9],frameData[10],frameData[11],frameData[8]]
		chestRotEnd = [frameDataNext[9],frameDataNext[10],frameDataNext[11],frameDataNext[8]]
		chestRot = p.getQuaternionSlerp(chestRotStart,chestRotEnd,frameFraction)
		
		neckRotStart = [frameData[13],frameData[14],frameData[15],frameData[12]]
		neckRotEnd= [frameDataNext[13],frameDataNext[14],frameDataNext[15],frameDataNext[12]]
		neckRot =  p.getQuaternionSlerp(neckRotStart,neckRotEnd,frameFraction)
		
		rightHipRotStart = [frameData[17],frameData[18],frameData[19],frameData[16]]
		rightHipRotEnd = [frameDataNext[17],frameDataNext[18],frameDataNext[19],frameDataNext[16]]
		rightHipRot = p.getQuaternionSlerp(rightHipRotStart,rightHipRotEnd,frameFraction)
		
		rightKneeRotStart = [frameData[20]]
		rightKneeRotEnd = [frameDataNext[20]]
		rightKneeRot = [rightKneeRotStart[0]+frameFraction*(rightKneeRotEnd[0]-rightKneeRotStart[0])]
		
		rightAnkleRotStart = [frameData[22],frameData[23],frameData[24],frameData[21]]
		rightAnkleRotEnd = [frameDataNext[22],frameDataNext[23],frameDataNext[24],frameDataNext[21]]
		rightAnkleRot =  p.getQuaternionSlerp(rightAnkleRotStart,rightAnkleRotEnd,frameFraction)
			
		rightShoulderRotStart = [frameData[26],frameData[27],frameData[28],frameData[25]]
		rightShoulderRotEnd = [frameDataNext[26],frameDataNext[27],frameDataNext[28],frameDataNext[25]]
		rightShoulderRot = p.getQuaternionSlerp(rightShoulderRotStart,rightShoulderRotEnd,frameFraction)
		
		rightElbowRotStart = [frameData[29]]
		rightElbowRotEnd = [frameDataNext[29]]
		rightElbowRot = [rightElbowRotStart[0]+frameFraction*(rightElbowRotEnd[0]-rightElbowRotStart[0])]
		
		leftHipRotStart = [frameData[31],frameData[32],frameData[33],frameData[30]]
		leftHipRotEnd = [frameDataNext[31],frameDataNext[32],frameDataNext[33],frameDataNext[30]]
		leftHipRot = p.getQuaternionSlerp(leftHipRotStart,leftHipRotEnd,frameFraction)
		
		leftKneeRotStart = [frameData[34]]
		leftKneeRotEnd = [frameDataNext[34]]
		leftKneeRot = [leftKneeRotStart[0] +frameFraction*(leftKneeRotEnd[0]-leftKneeRotStart[0]) ]
		
		leftAnkleRotStart = [frameData[36],frameData[37],frameData[38],frameData[35]]
		leftAnkleRotEnd = [frameDataNext[36],frameDataNext[37],frameDataNext[38],frameDataNext[35]]
		leftAnkleRot = p.getQuaternionSlerp(leftAnkleRotStart,leftAnkleRotEnd,frameFraction)
		
		leftShoulderRotStart = [frameData[40],frameData[41],frameData[42],frameData[39]]
		leftShoulderRotEnd = [frameDataNext[40],frameDataNext[41],frameDataNext[42],frameDataNext[39]]
		leftShoulderRot = p.getQuaternionSlerp(leftShoulderRotStart,leftShoulderRotEnd,frameFraction)
		leftElbowRotStart = [frameData[43]]
		leftElbowRotEnd = [frameDataNext[43]]
		leftElbowRot = [leftElbowRotStart[0]+frameFraction*(leftElbowRotEnd[0]-leftElbowRotStart[0])]
		
		#print("chestRot=",chestRot)
		p.setGravity(0,0,0)
		
		
		kp=1
			
		p.setJointMotorControlMultiDof(humanoid,chest,p.POSITION_CONTROL, targetPosition=chestRot,positionGain=kp, force=maxForce)
		p.setJointMotorControlMultiDof(humanoid,neck,p.POSITION_CONTROL,targetPosition=neckRot,positionGain=kp, force=maxForce)
		p.setJointMotorControlMultiDof(humanoid,rightHip,p.POSITION_CONTROL,targetPosition=rightHipRot,positionGain=kp, force=maxForce)
		p.setJointMotorControlMultiDof(humanoid,rightKnee,p.POSITION_CONTROL,targetPosition=rightKneeRot,positionGain=kp, force=maxForce)
		p.setJointMotorControlMultiDof(humanoid,rightAnkle,p.POSITION_CONTROL,targetPosition=rightAnkleRot,positionGain=kp, force=maxForce)
		p.setJointMotorControlMultiDof(humanoid,rightShoulder,p.POSITION_CONTROL,targetPosition=rightShoulderRot,positionGain=kp, force=maxForce)
		p.setJointMotorControlMultiDof(humanoid,rightElbow, p.POSITION_CONTROL,targetPosition=rightElbowRot,positionGain=kp, force=maxForce)
		p.setJointMotorControlMultiDof(humanoid,leftHip, p.POSITION_CONTROL,targetPosition=leftHipRot,positionGain=kp, force=maxForce)
		p.setJointMotorControlMultiDof(humanoid,leftKnee, p.POSITION_CONTROL,targetPosition=leftKneeRot,positionGain=kp, force=maxForce)
		p.setJointMotorControlMultiDof(humanoid,leftAnkle, p.POSITION_CONTROL,targetPosition=leftAnkleRot,positionGain=kp, force=maxForce)
		p.setJointMotorControlMultiDof(humanoid,leftShoulder, p.POSITION_CONTROL,targetPosition=leftShoulderRot,positionGain=kp, force=maxForce)
		p.setJointMotorControlMultiDof(humanoid,leftElbow, p.POSITION_CONTROL,targetPosition=leftElbowRot,positionGain=kp, force=maxForce)

		shift = [0.0, 0.0, 0.0]
		if actualSMPLPoses is not None:
			joint = actualSMPLPoses[frame]

			shift = np.array(shift)

			SMPLLinks = [(0, 2), (2, 5), (5, 8), (8, 11), 
				(0, 1), (1, 4), (4, 7), (7, 10), 
				(0, 3), (3, 6), (6, 9), (9, 12), (12, 15), 
				(9, 14), (12, 14), (14, 17), (17, 19), (19, 21), (21, 23),
				(9, 13), (13, 16), (16, 18), (18, 20), (20, 22)]
			for (i, j) in SMPLLinks:
				
				p.addUserDebugLine(lineFromXYZ=joint[i]+shift,
								lineToXYZ=joint[j]+shift,
								lineColorRGB=(0,0,0),
								lineWidth=1,
								lifeTime=0.1)


		p.stepSimulation()
	
	# time.sleep(100.)

	return


if __name__ == "__main__":
	thetas = hmrdata['thetas']
	cam = thetas[0][0:3]
	theta = thetas[0][3:75]
	beta = thetas[0][75:]
	path = "/home/gilles/Desktop/ORCV_Project/DeepMimic/data/motions/humanoid3d_cartwheel.txt"
	with open(path, 'r') as f:
		motion_dict = json.load(f)

	frames = np.array(motion_dict['Frames'])

	print(frames.shape)

	displayBulletFrames(frames)

