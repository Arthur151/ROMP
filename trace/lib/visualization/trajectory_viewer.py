from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

import json
import numpy as np
from PIL import Image, ImageOps
import numpy as np
import sys, math
import time
import _pickle as pickle
import timeit


global OGL_Settings
from .opengl_utils import initialize_opengl,  \
    setCameraView, setCameraViewOrth, setFree3DView, DrawBackgroundOrth, DrawBackground, \
    DrawSkeletons, DrawTrajectory, DrawMeshes, DrawPosOnly, DrawCameras, \
    RenderDomeFloor, RenderText, \
    reshape, keyboard, mouse, motion, specialkeys, ensure_fps, \
    SaveScenesToFile, OGL_Settings
    

def renderscene():
    start = timeit.default_timer()

    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)

    #Some anti-aliasing code (seems not working, though)
    glEnable (GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable (GL_LINE_SMOOTH)
    glHint (GL_LINE_SMOOTH_HINT, GL_NICEST)
    # glEnable(GL_POLYGON_SMOOTH)
    glEnable(GL_MULTISAMPLE)
    # glHint(GL_MULTISAMPLE_FILTER_HINT_NV, GL_NICEST)

    # Set up viewing transformation, looking down -Z axis
    glLoadIdentity()
    gluLookAt(0,0,0, 0, 0, 1, 0, -1, 0)

    if OGL_Settings['g_viewMode']=='camView':       #Set View Point in MTC Camera View
        if OGL_Settings['g_bOrthoCam']:
            setCameraViewOrth()
        else:
            setCameraView()

    else:#Free Mode
        # Set perspective (also zoom)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        #gluPerspective(zoom, float(OGL_Settings['g_Width'])/float(OGL_Settings['g_Height']), g_nearPlane, g_farPlane)
        gluPerspective(65, float(OGL_Settings['g_Width'])/float(OGL_Settings['g_Height']), OGL_Settings['g_nearPlane'], OGL_Settings['g_farPlane'])         # This should be called here (not in the reshpe)
        glMatrixMode(GL_MODELVIEW)
        setFree3DView()
        glColor3f(0,1,0)

    #This should be drawn first, without depth test (it should be always back)
    if OGL_Settings['g_bShowBackground']:
        if OGL_Settings['g_bOrthoCam']:
            DrawBackgroundOrth()
        else:
            DrawBackground()

    glEnable(GL_LIGHTING)
    glEnable(GL_CULL_FACE)

    # glUseProgram(0)
    glPolygonMode(GL_FRONT, GL_FILL)
    glPolygonMode(GL_BACK, GL_FILL)

    if OGL_Settings['g_bShowSkeleton']:
        DrawSkeletons()
        # DrawSkeletonsGT()
    #DrawTrajectory()
    if OGL_Settings['g_bShowMesh']:
        DrawMeshes()
    #DrawPosOnly()

    glDisable(GL_LIGHTING)
    glDisable(GL_CULL_FACE)

    #DrawCameras()

    if OGL_Settings['g_bShowFloor']:
        RenderDomeFloor()

    if OGL_Settings['g_show_fps']:
        RenderText("{0} fps".format(int(np.round(OGL_Settings['g_fps'],0))))

    # swap the screen buffers for smooth animation
    glutSwapBuffers()

    if OGL_Settings['g_bRotateView']:
        OGL_Settings['g_xRotate'] += OGL_Settings['g_rotateInterval']
        OGL_Settings['g_saveFrameIdx'] = OGL_Settings['g_rotateView_counter']
        OGL_Settings['g_rotateView_counter']+=1

    if OGL_Settings['g_bSaveToFile']:
        SaveScenesToFile(OGL_Settings, folderPath='trajectory_visual')

    OGL_Settings['g_frameIdx'] +=1
    if OGL_Settings['g_frameIdx'] >= OGL_Settings['g_frameLimit']:
        if OGL_Settings['g_bSaveOnlyMode']: # whether stop loop
            OGL_Settings['g_stopMainLoop']= True
        OGL_Settings['g_frameIdx'] = 0
    
    ensure_fps(start, fps=30)

def init_gl_util(OGL_Settings):
    g_lastframetime = g_currenttime = time.time()
    g_currenttime = time.time()
    refresh_fps = 0.15
    OGL_Settings['g_fps'] = (1-refresh_fps)*OGL_Settings['g_fps'] + refresh_fps*1/(g_currenttime-g_lastframetime)

    if OGL_Settings['g_bGlInitDone']==False:
        glutInit()
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH | GLUT_MULTISAMPLE) #GLUT_MULTISAMPLE is required for anti-aliasing
        #glutInitDisplayMode(GLUT_RGB |GLUT_DOUBLE|GLUT_DEPTH)
        glutInitWindowPosition(100,100)
        glutInitWindowSize(OGL_Settings['g_Width'],OGL_Settings['g_Height'])
        OGL_Settings['g_winID'] = glutCreateWindow("Visualize human skeleton")

        initialize_opengl(OGL_Settings)

        glutReshapeFunc(reshape)
        glutDisplayFunc(renderscene)
        glutKeyboardFunc(keyboard)
        glutMouseFunc(mouse)
        glutMotionFunc(motion)
        glutSpecialFunc(specialkeys)
        glutIdleFunc(renderscene)

        #Ver 2: for better termination (by pressing 'q')
        OGL_Settings['g_bGlInitDone'] = True
    else:
        glutReshapeWindow(OGL_Settings['g_Width'], OGL_Settings['g_Height']) 

def run_visualizer(OGL_Settings, maxIter=-10):
    # Setup for double-buffered display and depth testing
    init_gl_util(OGL_Settings)
    OGL_Settings['g_stopMainLoop']=False
    while True:
        # re-draw the image. 
        glutPostRedisplay()
        if bool(glutMainLoopEvent)==False:
            continue
        glutMainLoopEvent()
        if OGL_Settings['g_stopMainLoop']:
            break
        if maxIter>0:
            maxIter -=1
            if maxIter<=0:
                OGL_Settings['g_stopMainLoop'] = True

def prepare_data(data):
    frameLens = []
    OGL_Settings['VIS_DATA'] = {'Skeleton': None, 'Skeleton_GT': None, 'Mesh': None, 'BodyNormals':None, 'Trajectory':None, 'Position':None}
    if 'Skeleton' in data: ##Input: skel_list (skelNum, dim, frames): nparray or list of arrays (dim, frames)
        frameLens += [l.shape[1] for l in data['Skeleton']]
        OGL_Settings['VIS_DATA']['Skeleton'] = data['Skeleton']
    #Input: skel_list peopelNum x {'ver': vertexInfo, 'f': faceInfo}
    #: vertexInfo should be (frames x vertexNum x 3 ) #: faceInfo should be (vertexNum x 3 )
    if 'Mesh' in data: ##Input: skel_list (skelNum, dim, frames): nparray or list of arrays (dim, frames)
        frameLens += [l['ver'].shape[1] for l in data['Mesh']]
        OGL_Settings['VIS_DATA']['Mesh'] = data['Mesh']
    print(frameLens)
    OGL_Settings['g_frameLimit'] = max(OGL_Settings['g_frameLimit'],min(frameLens))

def show_trajectory(kp3ds, trajs=None):
    if trajs is not None:
        new_kp3ds = []
        for ind in range(len(kp3ds)):
            new_kp3ds.append(kp3ds[ind] + trajs[ind].unsqueeze(1))
            new_kp3ds[-1][kp3ds[ind]==-2.] = -2
        kp3ds = new_kp3ds
    Skeleton = []
    for kp3d in kp3ds:
        Skeleton_item = kp3d.numpy().reshape((-1, args().joint_num*3)).transpose((1,0))
        Skeleton.append(Skeleton_item)
    
    VIS_DATA = {'Skeleton':Skeleton}
    prepare_data(VIS_DATA)
    run_visualizer(OGL_Settings, maxIter=-10)

if __name__=='__main__':
    
    from .opengl_utils import loadBodyData
    seqName = "/home/yusun/Downloads/CMU_Panoptic/170915_toddler5"
    bodyData = loadBodyData(seqName)
    print('visualization person number:', len(bodyData), bodyData[0]['joints19'].shape)
    print(bodyData[0]['joints19'][:,-10:])
    #Set data on visualizer
    VIS_DATA = {'Skeleton':[bodyData[humanID]['joints19'] for humanID in range(len(bodyData))]}
    prepare_data(VIS_DATA)
    run_visualizer(OGL_Settings, maxIter=-10)

"""

#########################################################
#                    Setting data                       #
#########################################################

#bodyNormal_list: each element should have 3xframe
def setBodyNormal(bodyNormal_list):
    global g_bodyNormals#nparray: (faceNum, faceDim, frameNum)

    #g_bodyNormals  = bodyNormal_list
    g_bodyNormals  = [ x.copy() for x in bodyNormal_list]

    frameLens = [l.shape[1] for l in g_bodyNormals  if len(l)>0]
    maxFrameLength = max(frameLens)


    for i, p in enumerate(g_bodyNormals):
        if len(p)==0:
            newData = np.zeros((3, maxFrameLength))
            g_bodyNormals[i] = newData
        elif p.shape[0]==2:
            newData = np.zeros((3, p.shape[1]))
            newData[0,:] = p[0,:]
            newData[1,:] = 0 #some fixed number
            newData[2,:] = p[1,:]

            g_bodyNormals[i] = newData

        elif p.shape[0]==3:
            g_bodyNormals[i] = p

    OGL_Settings['g_frameLimit'] = max(OGL_Settings['g_frameLimit'],min(frameLens))


#os_list: each element should have 3xframe
#Input: skel_list (skelNum, dim, frames): 
def setPosOnly(pos_list):
    global g_posOnly,OGL_Settings['g_frameLimit']#nparray: (faceNum, faceDim, frameNum)

    g_posOnly  = [ x.copy() for x in pos_list]

    for i, p in enumerate(g_posOnly):
        if p.shape[0]==2:
            newData = np.zeros((3, p.shape[1]))
            newData[0,:] = p[0,:]
            #newData[1,:] = -100 #some fixed number
            newData[1,:] = 0 #some fixed number
            newData[2,:] = p[1,:]

            g_posOnly[i] = newData

    frameLens = [l.shape[1] for l in g_posOnly]
    OGL_Settings['g_frameLimit'] = max(OGL_Settings['g_frameLimit'],min(frameLens))


#Input: skel_list (skelNum, dim, frames): nparray or list of arrays (dim, frames)
def setTrajectory(traj_list):
    #Add Skeleton Data
    global g_trajectory,OGL_Settings['g_frameLimit'] #nparray: (skelNum, skelDim, frameNum)
    g_trajectory = traj_list #List of 2dim np.array
    frameLens = [l.shape[1] for l in g_trajectory]
    OGL_Settings['g_frameLimit'] = max(OGL_Settings['g_frameLimit'],min(frameLens))


#Input: skel_list (skelNum, dim, frames): nparray or list of arrays (dim, frames)
def setSkeleton(skel_list, bIsGT = False, jointType=None):
    global g_skeletons,OGL_Settings['g_frameLimit'] #nparray: (skelNum, skelDim, frameNum)
    global g_skeletons_GT,OGL_Settings['g_frameLimit'] #nparray: (skelNum, skelDim, frameNum)

    if bIsGT==False:
        global g_skeletonType
        g_skeletonType = jointType

    if jointType =='smpl':
        print("Use smplcoco instead of smpl!")
        assert(False)

    if isinstance(skel_list,list) == False and len(skel_list.shape)==2:
        skel_list = skel_list[np.newaxis,:]

    if bIsGT==False:
        #Add Skeleton Data
        g_skeletons = skel_list #List of 2dim np.array
    else:
         #Add Skeleton Data
        g_skeletons_GT = skel_list #List of 2dim np.array
    setFrameLimit()


#Input: skel_list peopelNum x {'ver': vertexInfo, 'f': faceInfo}
#: vertexInfo should be (frames x vertexNum x 3 )
#: if vertexInfo has (vertexNum x 3 ), this function automatically changes it to (1 x vertexNum x 3)
#: faceInfo should be (vertexNum x 3 )
#'normal': if missing, draw mesh by wireframes
def setMeshData(mesh_list, bComputeNormal = False):

    global g_meshes
    g_meshes = mesh_list

    if len(g_meshes)==0:
        return

    if len(g_meshes)>40:
        print("Warning: too many meshes ({})".format(len(g_meshes)))
        g_meshes =g_meshes[:40]

    for element in g_meshes:
        if len(element['ver'].shape) ==2:
            # print("## setMeshData: Warning: input size should be (N, verNum, 3). Current input is (verNum, 3). I am automatically fixing this.")
            element['ver'] = element['ver'][np.newaxis,:,:]
            if 'normal' in element.keys():
                element['normal'] = element['normal'][np.newaxis,:,:]

    #Auto computing normal
    if bComputeNormal:
        # print("## setMeshData: Computing face normals automatically.")
        for element in g_meshes:
            element['normal'] = ComputeNormal(element['ver'],element['f']) #output: (N, 18540, 3)

    setFrameLimit()

# Getting back to the original code
def setFrameLimit():
    OGL_Settings['g_frameLimit'] = 0

    if g_meshes is not None:
        frameLens = [l['ver'].shape[0] for l in g_meshes] 
        OGL_Settings['g_frameLimit'] = max(OGL_Settings['g_frameLimit'],min(frameLens))

    if g_skeletons is not None:
        frameLens = [l.shape[1] for l in g_skeletons] 
        OGL_Settings['g_frameLimit'] = max(OGL_Settings['g_frameLimit'],min(frameLens))

    if g_skeletons_GT is not None:
        frameLens = [l.shape[1] for l in g_skeletons_GT] 
        OGL_Settings['g_frameLimit'] = max(OGL_Settings['g_frameLimit'],min(frameLens))



#####################
#    show SMPL      #
#####################

def show_SMPL_sideView(bSaveToFile = False, bResetSaveImgCnt=True, countImg = True):
    show_SMPL_cameraView(bSaveToFile, bResetSaveImgCnt, countImg, False)

def show_SMPL_youtubeView(bSaveToFile = False, bResetSaveImgCnt=True, countImg = True, zoom = 230):
    show_SMPL(bSaveToFile = bSaveToFile, bResetSaveImgCnt = bResetSaveImgCnt, countImg = countImg, zoom = zoom, mode = 'youtube')

def show_SMPL_cameraView(bSaveToFile = False, bResetSaveImgCnt=True, countImg = True, bShowBG = True):
    show_SMPL(bSaveToFile = bSaveToFile, bResetSaveImgCnt = bResetSaveImgCnt, countImg = countImg, bShowBG = bShowBG, mode = 'camera')

# This is to render scene in camera view (especially MTC output)
def show_SMPL(bSaveToFile = False, bResetSaveImgCnt = True, countImg = True, bShowBG = True, zoom = 230, mode = 'camera'):
    init_gl_util()

    if mode == 'init':
        #Setup for rendering
        keyboard('c',0,0)

    global g_bSaveToFile, g_bSaveToFile_done, g_bShowSkeleton, OGL_Settings['g_bShowFloor'], g_viewMode, g_saveFrameIdx

    g_bSaveToFile_done = False
    g_bSaveToFile = bSaveToFile
    g_bShowSkeleton = False
    OGL_Settings['g_bShowFloor'] = False

    if mode == 'youtube':
        bShowBG = True
        global g_xTrans,  g_yTrans, g_zoom, g_xRotate, g_yRotate, g_zRotate
        if False:   #Original
            g_xTrans=  -86.0
            g_yTrans= 0.0
            g_zoom= zoom
            g_xRotate = 34.0
            g_yRotate= -32.0
            g_zRotate= 0.0
            g_viewMode = 'free'
        else:   # New
            g_xTrans=  -0.7
            g_yTrans= 3.86
            g_zoom= zoom #230
            g_xRotate = 29
            g_yRotate= -41
            g_zRotate= 0.0
            g_viewMode = 'free'
    elif mode == 'camera' or mode == 'init':
        g_viewMode = 'camView'

    OGL_Settings['g_bShowBackground'] = bShowBG

    if bResetSaveImgCnt:
        g_saveFrameIdx = 0 #+= 1      #always save as: scene_00000000.png
    elif countImg:
        g_saveFrameIdx +=1

    if mode == 'init':
        global g_stopMainLoop
        g_stopMainLoop=False
        # while True:
        while OGL_Settings['g_rotateView_counter']*OGL_Settings['g_rotateInterval']<360:
            glutPostRedisplay()
            if bool(glutMainLoopEvent)==False:
                continue
            glutMainLoopEvent()
            break
            if g_stopMainLoop:
                break
    else:
        if g_bSaveToFile:
            while g_bSaveToFile_done == False:
                glutPostRedisplay()
                if bool(glutMainLoopEvent)==False:
                    continue
                glutMainLoopEvent()
        else:
            for i in range(3):   ##Render more than one to be safer
                glutPostRedisplay()
                if bool(glutMainLoopEvent)==False:
                    continue
                glutMainLoopEvent()
"""