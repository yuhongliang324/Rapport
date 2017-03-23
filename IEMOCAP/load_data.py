__author__ = 'yuhongliang324'
import os
from scipy.io import loadmat


data_root = '/multicomp/users/liangke/MM/datasets/IEMOCAP/features'
# Order: Angry, Happy, Sad, Neutral
oldl_newl = {2: 0, 8: 1, 5: 2, 9: 3, 3: 1}


def load_feature(feature_name='audio'):
    sessions = os.listdir(data_root)
    sessions.sort()
    session_Xs = {}
    session_y = {}
    for session in sessions:
        if not session.startswith('Session'):
            continue
        print session
        sessionID = int(session[-1])
        session_path = os.path.join(data_root, session)
        Xs, y = load_feature_session(session_path, feature_name)
        session_Xs[sessionID] = Xs
        session_y[sessionID] = y
    sessionIDs = session_y.keys()
    len500, len1000, len2000, len3000 = 0, 0, 0, 0
    for sessionID in sessionIDs:
        Xs = session_Xs[sessionID]
        for X in Xs:
            if X.shape[0] > 500:
                len500 += 1
            if X.shape[0] > 1000:
                len1000 += 1
            if X.shape[0] > 2000:
                len2000 += 1
            if X.shape[0] > 3000:
                len3000 += 1
    print len500, len1000, len2000, len3000


def load_feature_session(session_path, feature_name):
    Xs = []
    y = []
    videos = os.listdir(session_path)
    videos.sort()
    for video in videos:
        video_path = os.path.join(session_path, video)
        if not os.path.isdir(video_path):
            continue
        mats = os.listdir(video_path)
        mats.sort()
        for mat in mats:
            if not mat.endswith('mat'):
                continue
            if mat == 'Ses04F_script02_1_M039.mat':
                print 'aa'
                continue
            mat_path = os.path.join(video_path, mat)
            data = loadmat(mat_path)
            label = data['emotion_label'][0][0]
            if label not in oldl_newl:
                continue
            label = oldl_newl[label]
            if feature_name == 'audio':
                X = data['audio_features']
            else:
                X = data['rotated_features']
            Xs.append(X)
            y.append(label)
    return Xs, y


def test1():
    load_feature()


if __name__ == '__main__':
    test1()
