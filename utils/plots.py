import cv2
import numpy as np
import random

def colors(index, bgr=False):
    palette = np.array([[255, 56, 56], [255, 157, 151], [255, 112, 31],
                        [255, 178, 29], [207, 210, 49], [72, 249, 10],
                        [146, 204, 23], [61, 219, 134], [26, 147, 52],
                        [0, 212, 187], [44, 153, 168], [0, 194, 255],
                        [52, 69, 147], [100, 115, 255], [0, 24, 236],
                        [132, 56, 255], [82, 0, 133], [203, 56, 255],
                        [255, 149, 200], [255, 55, 199]])
    c = palette[index % len(palette)]
    # 这里直接返回 Python int，不用担心 numpy.int64
    return (int(c[2]), int(c[1]), int(c[0])) if bgr else (int(c[0]), int(c[1]), int(c[2]))


def plot_one_box(x, img, color=(128, 128, 128), label=None, line_thickness=3):
    # 1. 强制把 color 转成纯 Python int tuple
    color = tuple(int(v) for v in color)
    # 2. 计算线宽
    tl = line_thickness or int(round(0.002 * max(img.shape[0:2])))
    # 3. 左上、右下坐标
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    # 4. 画框
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    # 5. 如果有标签，再画标签背景和文字
    if label:
        tf = max(tl - 1, 1)  # 字体粗细
        # 计算文字区域大小
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        # 标签背景的右下坐标
        c2_label = (c1[0] + t_size[0], c1[1] - t_size[1] - 3)
        # 画填充矩形
        cv2.rectangle(img, c1, c2_label, color, thickness=-1, lineType=cv2.LINE_AA)
        # 文字颜色也用纯 Python int tuple
        text_color = (225, 255, 255)
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, text_color,
                    thickness=tf, lineType=cv2.LINE_AA)
