import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2 as cv
import sys
from copy import deepcopy
sys.path.append("thirdparty/EdgeSAM")
from edge_sam import sam_model_registry, SamPredictor

sam_checkpoint = "weights/edge_sam_3x.pth"
model_type = "edge_sam"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device='cuda')
predictor = SamPredictor(sam)

box_mode = False
point_mode = True
positive = True
held_down = False
start = (0, 0)
points = []
labels = []
img = cv.imread("images/truck.jpg")
img = cv.resize(img, (640, 480))
orig_img = deepcopy(img)
point_img = deepcopy(img)
box = None

print("Generating image embedding")
predictor.set_image(img)

def show_mask(mask, img):
    color = np.array([50/255, 144/255, 255/255])
    segment = img[mask, :].astype(float) * color
    img[mask, :] = segment.astype(int)

def predict(predictor, points, labels, box):
    masks, scores, _ = predictor.predict(
        point_coords=np.array(points) if len(points) else None,
        point_labels=np.array(labels) if len(points) else None,
        box=box,
        num_multimask_outputs=4,
        use_stability_score = True
    )

    best_mask = masks[np.argmax(scores)]

    return best_mask

def draw_box(img, start, end):
    cv.rectangle(img, start, end, (0,255,0), 2)

def draw_points(img, points, labels):
    for i in range(len(points)):
        point = points[i]
        color = (0, 255, 0) if labels[i] else (0, 0, 0)
        cv.circle(img, point, 2, color, 2)

def on_click(event, x, y, flags, param):
    global box_mode, point_mode, held_down, start, img, predictor, positive, labels, box, point_img
    if event == cv.EVENT_LBUTTONDOWN:
        start = (x, y)
        held_down = True

    if event == cv.EVENT_MOUSEMOVE:
        if box_mode and held_down:
            img = deepcopy(point_img)
            draw_box(img, start, (x, y))

    if event == cv.EVENT_LBUTTONUP:
        held_down = False
        img = deepcopy(orig_img)

        if point_mode:
            points.append([x,y])
            labels.append(int(positive))

        if box_mode:
            box = np.array([*start, x, y])
            box = box[None, :]

        mask = predict(predictor, points, labels, box)
        show_mask(mask, img)

        draw_points(img, points, labels)
        point_img = deepcopy(img)

        if box is not None:
            draw_box(img, box[0][0:2], box[0][2:])


cv.namedWindow('image')
cv.setMouseCallback('image', on_click)

print("""Commands:
b: Box mode
p: Positive point mode
n: Negative point mode
c: Clear points
x: Clear box
esc: exit
""")
while True:
    cv.imshow('image', img)
    cmd = cv.waitKey(20) & 0xFF

    if cmd == ord('b'):
        print('Box Mode')
        box_mode = True
        point_mode = False
        #points = []
        #labels = []
        #img = deepcopy(orig_img)
    if cmd == ord('p'):
        print('Positive Point Mode')
        point_mode = True
        box_mode = False
        positive = True
    if cmd == ord('n'):
        print('Negative Point Mode')
        point_mode = True
        box_mode = False
        positive = False

    if cmd == ord('c'):
        print('Cleared Points')
        points = []
        labels = []

        img = deepcopy(orig_img)

        if box is not None:
            draw_box(img, box[0][0:2], box[0][2:])

            mask = predict(predictor, points, labels, box)
            show_mask(mask, img)
    if cmd == ord('x'):
        print('Cleared box')
        box = None
        img = deepcopy(orig_img)

        mask = predict(predictor, points, labels, box)
        show_mask(mask, img)

        draw_points(img, points, labels)
        point_img = deepcopy(img)



    if cmd == 27:
        break
