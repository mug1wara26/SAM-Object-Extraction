import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2 as cv
import os
from copy import deepcopy
import time
import sys

if len(sys.argv) != 3:
    print("Incorrect number of args supplied")
    print("python main.py <model_name> <path_to_image>")
    exit()
else:
    model_name = sys.argv[1]

    if model_name.lower() == "edge":
        sys.path.append("thirdparty/EdgeSAM")
        from edge_sam import sam_model_registry, SamPredictor
        sys.path.append("../..")

        sam_checkpoint = "weights/edge_sam_3x.pth"
        model_type = "edge_sam"
    elif model_name.lower() == "mobile":
        sys.path.append("thirdparty/MobileSAM")
        from mobile_sam import sam_model_registry, SamPredictor
        sys.path.append("../..")

        sam_checkpoint = "weights/mobile_sam.pt"
        model_type = "vit_t"
    else:
        print("Unsupported model type, currently only supports \"edge\" and \"mobile\"")


sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device='cuda' if torch.cuda.is_available() else 'cpu')
predictor = SamPredictor(sam)

box_mode = False
point_mode = True
positive = True
held_down = False
start = (0, 0)
points = []
labels = []

viz_size = (640, 480)
orig_img = cv.imread(sys.argv[2])
img = cv.resize(orig_img, viz_size)
point_img = deepcopy(img)
box = None
w_scale = orig_img.shape[1] / viz_size[0]
h_scale = orig_img.shape[0] / viz_size[1]

print("Generating image embedding")
predictor.set_image(orig_img)

def show_mask(mask, img):
    color = np.array([50/255, 144/255, 255/255])
    segment = img[mask, :].astype(float) * color
    img[mask, :] = segment.astype(int)

def predict(predictor, points, labels, box):
    start = time.time()

    scale_box = None
    if box is not None:
        scale_box = np.copy(box)
        scale_box[0][::2] = box[0][::2] * w_scale
        scale_box[0][1::2] = box[0][1::2] * h_scale

    # hack to parse conditional args to predict
    # EdgeSAM takes in num_multimask_outputs
    # MobileSAM takes in a boolean for multimask_output
    args = {}
    if sys.argv[1].lower() == "edge":
        args['num_multimask_outputs'] = 4
        args['use_stability_score'] = True
    if sys.argv[1].lower() == "mobile":
        args['multimask_output'] = True

    masks, scores, _ = predictor.predict(
        point_coords=(np.array(points) * [w_scale, h_scale]).astype(int) if len(points) else None,
        point_labels=np.array(labels) if len(points) else None,
        box=scale_box,
        **args
    )

    print(f"Inference time: {round(time.time() - start, 4)}s")

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
    # Should have probably started using classes 8 global variables ago
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
        tmp = deepcopy(orig_img)

        if point_mode:
            points.append([x,y])
            labels.append(int(positive))

        if box_mode:
            box = np.array([*start, x, y])
            box = box[None, :]

        mask = predict(predictor, points, labels, box)
        show_mask(mask, tmp)

        img = cv.resize(tmp, viz_size)

        draw_points(img, points, labels)
        point_img = deepcopy(img)

        if box is not None:
            draw_box(img, box[0][0:2], box[0][2:])


window_name = sys.argv[1].upper()
cv.namedWindow(window_name)
cv.setMouseCallback(window_name, on_click)

print("""Commands:
b: Box mode
p: Positive point mode
n: Negative point mode
c: Clear points
x: Clear box
esc: exit
""")
while True:
    cv.imshow(window_name, img)
    cmd = cv.waitKey(20) & 0xFF

    if cmd == ord('b'):
        print('Box Mode')
        box_mode = True
        point_mode = False
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

        tmp = deepcopy(orig_img)
        point_img = cv.resize(orig_img, viz_size)

        if box is not None:
            mask = predict(predictor, points, labels, box)
            show_mask(mask, tmp)

            scale_box = np.copy(box)
            scale_box[0][::2] = box[0][::2] * w_scale
            scale_box[0][1::2] = box[0][1::2] * h_scale
            scale_box.astype(int)

            draw_box(tmp, scale_box[0][0:2], scale_box[0][2:])

        img = cv.resize(tmp, viz_size)

    if cmd == ord('x'):
        print('Cleared box')
        box = None
        tmp = deepcopy(orig_img)

        mask = predict(predictor, points, labels, box)
        show_mask(mask, tmp)

        img = cv.resize(tmp, viz_size)

        draw_points(img, points, labels)
        point_img = deepcopy(img)

    if cmd == ord('s'):
        if len(points) != 0 or box is not None:
            print('Saving image to segments/')

            mask = predict(predictor, points, labels, box)
            rgba = cv.cvtColor(orig_img, cv.COLOR_RGB2RGBA)
            segment = rgba[mask, :]
            output_img = np.zeros_like(rgba)
            row_coords, col_coords = np.nonzero(mask)
            output_img[row_coords, col_coords] = rgba[row_coords, col_coords]

            rmin = min(row_coords)
            rmax = max(row_coords)
            cmin = min(col_coords)
            cmax = max(col_coords)


            cv.imwrite(f"segments/{len(os.listdir('segments'))}.png", output_img[rmin:rmax+1, cmin:cmax+1])


    if cmd == 27:
        break
