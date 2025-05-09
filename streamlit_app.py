# streamlit_app.py
import random

import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import time
from glob import glob

from PIL import Image

from backend import load_model, detect_video_frame, crop_express_image
from utils.clean import check_img_size

# 页面配置
st.set_page_config(layout="wide")
st.title("🚀 快递单版面检测演示")

# —— 侧边栏：模型选择 & 参数调整 —— #
st.sidebar.header("设置")

# 预设模型选项
model_options = {
    "v1": "./models/kuaidi_20250224_v1_best.onnx",
    "v2": "./models/kuaidi_20250227_best.onnx",
    "v3": "./models/kuaidi_20250310_v3_best.onnx",
}
model_choice = st.sidebar.selectbox("选择 ONNX 模型", list(model_options.keys()))

if model_options[model_choice] is None:
    custom_path = st.sidebar.text_input("自定义 ONNX 模型路径", "")
    model_path = custom_path.strip() or None
else:
    model_path = model_options[model_choice]

# 检测参数
conf_thres = st.sidebar.slider("置信度阈值", 0.0, 1.0, 0.5, 0.01)
iou_thres  = st.sidebar.slider("NMS IOU 阈值", 0.0, 1.0, 0.45, 0.01)
img_size   = st.sidebar.number_input("输入尺寸（正方形）", 64, 1280, 640, step=32)
max_det    = st.sidebar.number_input("最大检测数", 1, 10, 3, step=1)

# 视频源选择：上传或示例
st.sidebar.header("选择视频来源")
video_source = st.sidebar.radio("", ["上传视频", "示例视频"])

if video_source == "上传视频":
    uploaded = st.file_uploader("上传视频文件", type=["mp4", "avi", "mov"])
    video_path = None
else:
    sample_dir = r"./resource/video"
    sample_files = glob(os.path.join(sample_dir, "*.*"))
    sample_names = [os.path.basename(f) for f in sample_files]
    sample_choice = st.sidebar.selectbox("选择示例视频", sample_names)
    video_path = os.path.join(sample_dir, sample_choice)
    uploaded = None

start_btn = st.sidebar.button("开始离线检测并播放")

# Session State 缓存
if "processed_frames" not in st.session_state:
    st.session_state.processed_frames = []
    st.session_state.frame_results   = []
    st.session_state.fps             = None

# 离线检测逻辑
if start_btn:
    # 选择视频源
    if video_source == "上传视频":
        if not uploaded:
            st.error("请先上传视频文件")
            st.stop()
        t_in = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        t_in.write(uploaded.read())
        t_in.flush()
        t_in.close()
        cap = cv2.VideoCapture(t_in.name)
    else:
        cap = cv2.VideoCapture(video_path)

    # 校验模型路径
    if not model_path:
        st.error("⚠️ 请选择或输入有效的模型路径")
        cap.release()
        st.stop()

    # 加载模型
    model = load_model(model_path)
    stride = 32

    fps   = cap.get(cv2.CAP_PROP_FPS) or 10
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    # 重置缓存
    st.session_state.fps = fps
    st.session_state.processed_frames.clear()
    st.session_state.frame_results.clear()

    progress_bar = st.sidebar.progress(0)
    status_text  = st.sidebar.empty()

    # 批量检测
    for idx in range(total):
        ret, frame = cap.read()
        if not ret:
            break
        annotated, result = detect_video_frame(
            model, frame,
            img_size=check_img_size(img_size, s=stride),
            model_stride=stride,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            max_det=max_det
        )
        st.session_state.processed_frames.append(annotated)
        st.session_state.frame_results.append(result)

        progress_bar.progress((idx + 1) / total)
        status_text.text(f"已处理帧：{idx+1}/{total}")

    cap.release()
    status_text.text("✅ 离线检测完成，开始播放")

    # 删除临时文件（上传模式）
    if video_source == "上传视频":
        try:
            os.unlink(t_in.name)
        except:
            pass


#播放区

if st.session_state.processed_frames:
    # 1. 先创建一次两列布局和占位符
    video_col, info_col = st.columns([4, 1])
    video_slot = video_col.empty()
    info_slot  = info_col.empty()

    display_width = 600

    # 2. 然后再循环播放，每次只更新占位符内容
    for frame, result in zip(st.session_state.processed_frames, st.session_state.frame_results):
        # 更新视频帧
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_slot.image(img, width=display_width)

        # 更新右侧信息：把所有目标信息合成一段 HTML/Markdown，一次性更新
        if result:
            md = ""
            for o in result:
                x1, y1, x2, y2 = o["position"]
                md += f"""
<div style="background-color:#f8f9fa;border:1px solid #dee2e6;border-radius:4px;padding:8px;margin-bottom:8px;">
  <div style="font-size:35px;font-weight:bold;color:#343a40;">类别</div>
  <div style="font-size:30px;font-weight:bold;color:#113a40;text-align: center;">{o['label']}</div>
  <div style="font-size:35px;color:#155724;">置信度:</div>
  <div style="font-size:35px;color:#155724;text-align: center;">{o['conf']:.2f}</div>
  <div style="font-size:30px;color:#856404;">坐标:</div>
  <div style="font-size:30px;color:#856404;text-align: center;">({x1}, {y1}), ({x2}, {y2})</div>
</div>
"""
            info_slot.markdown(md, unsafe_allow_html=True)
        else:
            info_slot.markdown(
                '<div style="background-color:#f8d7da;padding:8px;border-radius:4px;">'
                '<span style="font-size:16px;color:#721c24;">本帧未检测到目标</span>'
                '</div>',
                unsafe_allow_html=True
            )

        # cropped = crop_express_image(frame, result)
        # # 转成 RGB 并展示
        # cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        # info_col.image(cropped_rgb, caption="裁剪后的快递单区域", use_container_width=True)

        # 控制帧率，让更新看起来连贯
        time.sleep(1.0 / st.session_state.fps)

    # 播放结束提示
    st.toast("检测已完成！", icon="✅")

#########################
all_crops = []
for frame, results in zip(st.session_state.processed_frames, st.session_state.frame_results):
    for det in results:
        # 拿最高置信度的“valid area”做裁剪
        if det["label"] == "valid area":
            crop = crop_express_image(frame, [det])
            all_crops.append((crop, det["conf"]))
# 按置信度排序，取前三
top30 = sorted(all_crops, key=lambda x: x[1], reverse=True)[:30]
n = min(3, len(top30))
if n > 0:
    top3 = random.sample(top30, n)
else:
    top3 = []
# 在播放区下方，展示三张裁剪图
top3_paths = []
if top3:
    st.markdown("### 📦 置信度最高的三张裁剪结果")
    cols = st.columns(3)
    for (img_crop, conf), col in zip(top3, cols):
        # BGR→RGB
        img_rgb = cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB)
        col.image(img_rgb, use_container_width=True)
        col.caption(f"置信度: {conf:.2f}")

        # 保存图片为临时文件
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        img_pil = Image.fromarray(img_rgb)
        img_pil.save(tmp_file.name)
        top3_paths.append(tmp_file.name)

# 保存到 session_state
st.session_state.top3_image_paths = top3_paths
#########################
