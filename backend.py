import base64
import os
import ollama
import numpy as np
import torch
import cv2
from cnocr import CnOcr
from PIL import Image, ImageDraw, ImageFont
from onnxruntime import InferenceSession
from ultralytics import YOLO
import streamlit as st
from utils.clean import letterbox, check_img_size, non_max_suppression, scale_coords
from utils.plots import plot_one_box, colors
from utils.model import UVDocnet
import torch.nn.functional as F

def load_model(model_path):
    return InferenceSession(model_path, providers=["CPUExecutionProvider"])

@torch.no_grad()
def detect_video_frame(model, frame, img_size=640, model_stride=32, conf_thres=0.5, iou_thres=0.45,
                       max_det=1000, agnostic_nms=False, classes=None, class_keys=["valid area"]):
    img_size = check_img_size(img_size, s=32)
    img0 = frame.copy()

    img = letterbox(img0, img_size, stride=model_stride)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
    img = np.ascontiguousarray(img).astype(np.float32) / 255.0
    if len(img.shape) == 3:
        img = np.expand_dims(img, axis=0)

    pred = model.run(None, {"images": img})[0]
    pred = torch.from_numpy(pred)
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

    result = []
    for det in pred:
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
            for *xyxy, conf, cls in reversed(det):
                c = int(cls)
                position = [int(x.item()) for x in xyxy]
                label = class_keys[c] if c < len(class_keys) else f"class_{c}"
                result.append({"position": position, "label": label, "conf": float(conf)})

                # draw box
                plot_one_box(xyxy, img0, label=f"{label} {conf:.2f}", color=colors(c, True), line_thickness=2)

    return img0, result


def crop_express_image(image, detections):
    if not detections:
        return image

    # 按照置信度从高到低排序检测结果
    detections.sort(key=lambda x: x["conf"], reverse=True)

    # 选择置信度最高的快递单
    for detection in detections:
        if detection["label"] == "valid area":
            x1, y1, x2, y2 = detection["position"]
            cropped_image = image[y1:y2, x1:x2]
            return cropped_image
    return image


################

IMG_SIZE = [488, 712]
GRID_SIZE = [45, 31]
def load_uvdoc_model(ckpt_path):

    model = UVDocnet(num_filter=32, kernel_size=5)
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt["model_state"])
    return model


def bilinear_unwarping(warped_img, point_positions, img_size):

    upsampled_grid = F.interpolate(
        point_positions, size=(img_size[1], img_size[0]), mode="bilinear", align_corners=True
    )
    unwarped_img = F.grid_sample(warped_img, upsampled_grid.transpose(1, 2).transpose(2, 3), align_corners=True)

    return unwarped_img

def unwarp_img(ckpt_path, img_path, img_size, model=None):
    device = torch.device("cpu")
    # 加载模型
    if model is None:
        model = load_uvdoc_model(ckpt_path)
    model.to(device)
    model.eval()

    # 加载图片
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    inp = torch.from_numpy(cv2.resize(img, img_size).transpose(2, 0, 1)).unsqueeze(0)

    # 调用模型预测
    inp = inp.to(device)
    point_positions2D, _ = model(inp)

    # 展平
    size = img.shape[:2][::-1]
    unwarped = bilinear_unwarping(
        warped_img=torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(device),
        point_positions=torch.unsqueeze(point_positions2D[0], dim=0),
        img_size=tuple(size),
    )
    unwarped = (unwarped[0].detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)

    # 保存结果
    unwarped_BGR = cv2.cvtColor(unwarped, cv2.COLOR_RGB2BGR)
    save_path = os.path.splitext(img_path)[0] + "_unwarp.png"
    cv2.imwrite(save_path, unwarped_BGR)
    return save_path

def perform_ocr(ocr_prompt, processed_img_path):
    try:
        if ocr_prompt:
            with open(processed_img_path, "rb") as f:
                image_data = f.read()
            image_b64 = base64.b64encode(image_data).decode("utf-8")

            # response = ollama.chat(
            #     model='granite3.2-vision',
            #     messages=[{
            #         'role': 'user',
            #         'content': ocr_prompt,
            #         'images': [image_b64]
            #     }]
            # )
            response = ollama.chat(
                model='minicpm-v:latest',
                messages=[{
                    'role': 'user',
                    'content': ocr_prompt,
                    'images': [image_b64]
                }]
            )

            return response.message.content
    except Exception as e:
        return f"OCR Error: {e}"

# def run_cnocr(img_path):
#     cv_img = cv2.imread(img_path)
#     if cv_img is None:
#         raise FileNotFoundError(f"无法读取图像文件：{img_path}")
#
#     # 初始化CnOcr
#     ocr = CnOcr(
#         rec_model_fp="./weights/ocr_model/ch_PP-OCRv4_det_infer.onnx",
#         det_model_fp="./weights/ocr_model/ocr_densenet.onnx"
#     )
#     results = ocr.ocr(img_path)
#
#     #转换为 PIL 图像以便绘制中文
#     img_pil = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
#     draw = ImageDraw.Draw(img_pil)
#
#
#     font_path = r'./fonts/simhei.ttf'
#     font = ImageFont.truetype(font_path, 20)
#
#     text_results = []
#     box_results = []
#     #绘制,记录结果
#     for res in results:
#         text = res['text']
#         box = res['position']  # [[x0, y0], ..., [x3, y3]]
#         polygon = [tuple(pt) for pt in box]
#         color = tuple(np.random.randint(0, 255, 3).tolist())
#
#         # 画框
#         draw.line(polygon + [polygon[0]], fill=color, width=2)
#         x, y = polygon[0]
#         draw.text((x, y - 22), text, fill=color, font=font)
#
#         text_results.append(text)
#         box_results.append(box)
#
#     #转回OpenCV
#     final_img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
#
#     return final_img, text_results,box_results

def run_cnocr_visualization(img_fp,
                            rec_model_fp=r'./weights/ocr_model/ocr_densenet.onnx',
                            det_model_fp=r'./weights/ocr_model/ch_PP-OCRv4_det_infer.onnx',
                            font_path=r'./fonts/simhei.ttf',
                            font_size=20):
    """
    用 CnOcr 对图像进行文字检测与识别，并在图像上绘制检测框和文字。
    参数：
      - img_fp:      待处理图像文件路径
      - rec_model_fp:OCR 识别模型 onnx 文件路径
      - det_model_fp:OCR 检测模型 onnx 文件路径
      - font_path:   用于绘制中文的 .ttf 字体文件路径
      - font_size:   绘制文字的字号
    返回：
      - final_img:   处理后并绘制的 BGR 格式 OpenCV 图像
      - text_results:列表，每个元素是识别出的文字
      - box_results: 列表，每个元素是对应的四点坐标 [[x0,y0],...,[x3,y3]]
    """
    # 1. 读取图像
    cv_img = cv2.imread(img_fp)
    if cv_img is None:
        raise FileNotFoundError(f"无法读取图像文件：{img_fp}")

    # 2. 初始化 OCR
    ocr = CnOcr(rec_model_fp=rec_model_fp, det_model_fp=det_model_fp)

    # 3. 执行 OCR
    results = ocr.ocr(img_fp)

    # 4. 转为 PIL 以绘制中文
    img_pil = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    # 5. 加载字体
    if not font_path or not font_path.endswith('.ttf'):
        raise FileNotFoundError(f"字体文件不存在或格式不支持：{font_path}")
    font = ImageFont.truetype(font_path, font_size)

    text_results = []
    box_results  = []

    # 6. 绘制
    for res in results:
        text = res['text']
        box  = res['position']  # [[x0,y0],...[x3,y3]]
        poly = [tuple(pt) for pt in box]
        color = tuple(np.random.randint(0, 255, 3).tolist())

        # 画框
        draw.line(poly + [poly[0]], fill=color, width=2)

        # 写字（在第一个点上方）
        x0, y0 = poly[0]
        draw.text((x0, y0 - font_size - 2), text, fill=color, font=font)

        text_results.append(text)
        box_results.append(box)

    # 7. 转回 OpenCV BGR
    final_img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    return final_img, text_results, box_results

def load_yolo_model(model_path):
    try:
        return YOLO(model_path)
    except Exception as e:
        st.error(f"YOLO模型加载失败: {e}")
        return None


def plot_results(cv_img, results):
    # 获取分类结果
    probs = results[0].probs  # 获取概率结果
    top1_label = results[0].names[probs.top1]  # 获取最高概率类别
    top1_conf = probs.top1conf.item()  # 获取置信度

    # 在图像中心添加文字
    text = f"{top1_label}:{top1_conf:.2f} "
    font_scale = 2
    thickness = 4

    # 计算文字位置
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
    text_x = (cv_img.shape[1] - text_size[0]) // 2
    text_y = (cv_img.shape[0] + text_size[1]) // 2

    # 添加背景
    cv2.rectangle(cv_img,
                  (text_x - 10, text_y - text_size[1] - 10),
                  (text_x + text_size[0] + 10, text_y + 10),
                  (255, 255, 255), -1)

    if top1_conf > 0.7:
        cv2.putText(cv_img, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)
    else:
        cv2.putText(cv_img, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), thickness - 1)
    return cv_img
