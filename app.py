import numpy as np
from flask import Flask, render_template
from flask_socketio import SocketIO, emit

app = Flask(__name__)
socketio = SocketIO(app)

# 4 states of ['idle', 'detect', 'pause', 'response', 'exit']
state = 'idle'
is_detected = False
is_changed = False
is_sound = False
lessons = []
card_index = 0
image = np.full((568, 945, 3), 255, np.uint8)
    
@socketio.on('connect')
def connect():
    print "client connected"

@socketio.on('registration')
def register(data):
    print 'registeration'
    emit('register', data)

@socketio.on('sayIntroduction')
def sayIntroduction(data):
    print 'say introduction'

@socketio.on('prevLesson')
def prevLesson(data):
    global state, card_index
    if state in ['response']:
        state = 'detect'
    card_index = max(card_index - 1, 0)
    update_card()
    print "Robot: prev lesson"

@socketio.on('nextLesson')
def nextLesson(data):
    global state, card_index
    if state in ['response']:
        state = 'detect'
    card_index = min(card_index + 1, len(lessons) - 1)
    update_card()
    print "Robot: next lesson"

@socketio.on('repeatLesson')
def repeatLesson(data):
    global state, card_index
    if state in ['response']:
        state = 'detect'
    card_index = 0
    update_card()
    print "Robot: repeat lesson"

@socketio.on('playLesson')
def playLesson(data):
    global state
    if state in ['pause']:
        state = 'detect'
    print "Robot: play lesson"

@socketio.on('pauseLesson')
def pauseLesson(data):
    global state
    if state in ['detect']:
        state = 'pause'
    print "Robot: pause lesson"

@socketio.on('positiveFB')
def positiveFB(data):
    global state, is_detected
    if state in ['idle', 'detect', 'pause']:
        state = 'response'
        is_detected = True
    print "Robot: positive"

@socketio.on('encourageFB')
def encourageFB(data):
    global state, is_detected
    if state in ['idle', 'detect', 'pause']:
        state = 'response'
        is_detected = False
    print "Robot: positive"

@socketio.on('assigns')
def assigns(data):
    global state, lessons
    lessons = []
    lessons_path = os.path.join(root, 'assets/lessons/images')
    folders = sorted(os.listdir(lessons_path))
    for d in data['payload'].split('_'):
        lesson_name = folders[int(d) - 1]
        cards_path = os.path.join(root, 'assets/lessons/images', lesson_name)
        cards = sorted(os.listdir(cards_path))
        for card in cards:
            card_path = os.path.join(lesson_name, card)
            lessons.append(card_path)
    # if state in ['detect', 'pause', 'response']:
    #     state = 'detect'
    if len(lessons) == 0:
        state = 'idle'
    else:
        repeatLesson(0)
    print('Robot:', lessons, 'assigned')

@socketio.on('disconnect')
def disconnect():
    print 'Client disconnected'

from threading import Thread
import time

class ThreadedSocket(object):
    def __init__(self, src=0):
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        socketio.run(app)
            

import os
import cv2
import time
import random
# import numpy as np
from subprocess import Popen
import pygame
pygame.mixer.init()
pygame.mixer.music.set_volume(1)

# Get project path
root = os.path.dirname(os.path.abspath(__file__))

# Initiate SIFT detector
sift = cv2.SIFT_create()

scale = 0.4

# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary
flann = cv2.FlannBasedMatcher(index_params,search_params)

def video_length():
    global v_len
    for folder in ['Correct', 'Encourage', 'Farewell', 'Greeting', 'IntoNextLesson', 'Waiting']:
        videos = os.listdir(os.path.join(root, 'assets/interfaces', folder))
        print('videos', videos)
        for v in videos:
            cap = cv2.VideoCapture(os.path.join(root, 'assets/interfaces', folder, v))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps
            v_len[v] = duration
v_len = dict()
video_length()

def play_video(action):
    global state

    blank = np.full((360, 640, 3), 255, np.uint8)
    cv2.imshow('window', blank)
    cv2.waitKey(1)

    video_path = os.path.join(root, 'assets/interfaces', action)
    video_name = random.choice(os.listdir(video_path))
    duration = v_len[video_name]

    # play video by command
    os.system('killall omxplayer.bin')
    omxc = Popen(['omxplayer', '-b', os.path.join(video_path, video_name)])
    time.sleep(duration)
    # state = 'detect' if len(lessons) else "idle"

def update_card():
    global image, is_sound, is_changed
    # load image
    image_path = os.path.join(root, 'assets/lessons/images', lessons[card_index])
    image = cv2.imread(image_path)
    # load sound (if available)
    sound_path = os.path.join(root, 'assets/lessons/sounds', lessons[card_index])
    sound_path = sound_path[:-4] + '.mp3'
    is_sound = os.path.isfile(sound_path)
    if is_sound: pygame.mixer.music.load(sound_path)
    # lesson has been changed
    is_changed = True
    

# main app
if __name__ == '__main__':

    # listen to system
    threaded_socket = ThreadedSocket()
    
    # set window fullscreen
    # cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
    # cv2.setWindowProperty("window",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
    
    # prepare camera + config
    cap = cv2.VideoCapture(0)
    idle_path = root + '/assets/interfaces/idle1.jpg' 
    idle = cv2.imread(idle_path)

    # default lessons 'Body_th'
    assigns({'payload': '10'})
    
    # play introduction
    play_video('IntoNextLesson')
    state = "detect"
    
    # loop lessons
    while True:

        print("state", state)

        # just kookkai face
        if state == 'idle':
            cv2.imshow('window', idle)
            key = cv2.waitKey(10)
            if key in [ord('q')]:
                state = 'exit'

        # hold card
        if state == 'pause':
            cv2.imshow('window', image)
            key = cv2.waitKey(10)
            if key in [ord('q')]:
                state = 'exit'

        # detect using corner detection     
        if state == 'detect':
            cv2.imshow('window', image)
            key = cv2.waitKey(500)

            # create template
            img1 = image.copy()
            img1 = cv2.resize(img1, (0, 0), fx=scale, fy=scale)
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) 
            kp1, des1 = sift.detectAndCompute(gray1, None)

            if is_sound:
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    key = cv2.waitKey(1)
            
            start = time.time()
            while state == 'detect':
                ret, camera = cap.read()
                img2 = camera.copy()
                if not ret: continue
                h, w, c = np.shape(img2)
                x1 = int(0.0 * w)
                x2 = int(0.9 * w)
                y1 = int(0.2 * h)
                y2 = int(0.9 * h)
                # img2 = img2[y1:y2, x1:x2]
                img2 = cv2.resize(img2, (0, 0), fx=scale, fy=scale)
                gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) 
                kp2, des2 = sift.detectAndCompute(gray2, None)

                score = 0.
                matches = flann.knnMatch(des1, des2, k=2)
                for i, (m,n) in enumerate(matches):
                    if m.distance < 0.7*n.distance:
			score += 1
                score /= len(des1)

                # detect something
                if score > 0.19:
                    state = 'response'
                    is_detected = True
                # not detect anything
                if time.time() - start > 7 and state == 'detect':
                    state = 'response'
                    is_detected = False
                # change lesson
                if is_changed:
                    break
                
                # cv2.imshow("camera", camera)
                # cv2.imshow("img2", img2)
                key = cv2.waitKey(1)
                if key in [ord('q')]:
                    state = 'exit'

            is_changed = False

        # response
        if state == 'response':
            h, w, c = np.shape(image)
            bordered_image = np.zeros((h, w, 3), np.uint8)
            bordered_image[:] = (0, 255, 0) if is_detected else (0, 0, 255)
            bordered_image[20:-20, 20:-20] = image[20:-20, 20:-20]
            cv2.imshow('window', bordered_image)
            cv2.waitKey(400)
            cv2.imshow('window', image)
            cv2.waitKey(400)
            cv2.imshow('window', bordered_image)
            cv2.waitKey(400)
            cv2.imshow('window', image)
            cv2.waitKey(400)
            cv2.imshow('window', bordered_image)
            cv2.waitKey(400)
            cv2.imshow('window', image)
            cv2.waitKey(400)

            if is_detected:
                play_video('Correct')
            else:
                play_video('Encourage')

            nextLesson(0)

        if state == 'exit':
            break
