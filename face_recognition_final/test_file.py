# In[]:
import cv2
import dlib
import numpy as np
from imutils import face_utils
import tensorflow as tf
import pickle
import onnx
import onnxruntime as ort
from onnx_tf.backend import prepare
import time
import numexpr as ne
from utility import predict
import imutils
# In[]:
def alignFace(box):
    x1, y1, x2, y2 = box
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aligned_face = fa.align(frame, gray, dlib.rectangle(left = x1, top=y1, right=x2, bottom=y2))
    aligned_face = cv2.resize(aligned_face, (112,112))
    aligned_face = aligned_face - 127.5
    aligned_face = aligned_face * 0.0078125
    return aligned_face

# In[]:
onnx_path = 'models/ultra_light/ultra_light_models/ultra_light_320.onnx'
onnx_model = onnx.load(onnx_path)
predictor = prepare(onnx_model)
ort_session = ort.InferenceSession(onnx_path)
input_name = ort_session.get_inputs()[0].name

shape_predictor = dlib.shape_predictor('models/facial_landmarks/shape_predictor_5_face_landmarks.dat')
fa = face_utils.facealigner.FaceAligner(shape_predictor, desiredFaceWidth=112, desiredLeftEye=(0.3, 0.3))

threshold = 0.5
# In[]:
# load distance
with open("embeddings/embeddings.pkl", "rb") as f:
    (saved_embeds, names) = pickle.load(f)
names = np.array(names)
# In[]:
with tf.Graph().as_default():
    with tf.Session() as sess:

        saver = tf.train.import_meta_graph('models/mfn/m1/mfn.ckpt.meta')
        saver.restore(sess, 'models/mfn/m1/mfn.ckpt')

        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        embedding_size = embeddings.get_shape()[1]
        
        video_capture = cv2.VideoCapture('faces/test/2.mkv')
        index=0
        start=time.time()
        while True:
            index+=1
            ret, frame = video_capture.read()
            frame = imutils.resize(frame, width=950)
            # preprocess faces for detection
            h, w, _ = frame.shape
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (320, 240))
            img_mean = np.array([127, 127, 127])
            img = (img - img_mean) / 128
            img = np.transpose(img, [2, 0, 1])
            img = np.expand_dims(img, axis=0)
            img = img.astype(np.float32)
            
            # detect faces
            confidences, boxes = ort_session.run(None, {input_name: img})
            boxes, labels, probs = predict(w, h, confidences, boxes, 0.7)

            # locate faces and pre-processing for embeddings ???
            boxes[boxes<0] = 0
            faces = list(map(alignFace, boxes))

            # face embedding
            if len(faces)>0:
                faces = np.array(faces)
                feed_dict = { images_placeholder: faces, phase_train_placeholder:False }
                embeds = sess.run(embeddings, feed_dict=feed_dict)

                # prediciton using distance 
                embeds_3D = embeds[:, np.newaxis]
                diff = ne.evaluate('saved_embeds-embeds_3D')
                dist = np.sum(np.square(diff),2)
                idx = np.argmin(dist,axis=1)
                predictions = np.array(['unknown            ']*len(idx))
                indexes = dist[range(len(idx)),idx]<threshold
                n_idx = idx[indexes]
                predictions[indexes] = names[n_idx]
                # draw
                for i, box in enumerate(boxes):
                    text = f"{predictions[i]}"
                    x1, y1, x2, y2 = box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (80,18,236), 2)
                    # Draw a label with a name below the face
                    cv2.rectangle(frame, (x1, y2 - 10), (x2, y2), (80,18,236), cv2.FILLED)
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(frame, text, (x1 + 6, y2 - 3), font, 0.3, (255, 255, 255), 1)
            
            fps=index/(time.time()-start)
            cv2.putText(frame, '%.2f'%fps,(0,50), cv2.FONT_HERSHEY_DUPLEX, 0.3, (255, 255, 255), 1)
                    
            cv2.imshow('Video', frame)

            # Hit 'q' on the keyboard to quit!
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

# Release handle to the webcam
#video_capture.release()
cv2.destroyAllWindows()
# In[]:
