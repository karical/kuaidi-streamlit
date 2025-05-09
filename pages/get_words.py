import streamlit as st
from PIL import Image
import os
import io
import numpy as np
import cv2
import tempfile
import torch

from backend import (
    load_uvdoc_model, unwarp_img,
    load_yolo_model, plot_results,
    perform_ocr,run_cnocr_visualization
)

# 常量
OCR_PROMPT_DEFAULT = "请识别快递单上的关键信息，例如快递单号、寄件人、收件人地址等。"
IMG_SIZE = [488, 712]

st.set_page_config(layout="wide")
st.title("📄图片处理与信息提取")

# 1️⃣ 选择图片
if "top3_image_paths" not in st.session_state or not st.session_state.top3_image_paths:
    st.warning("请先在主页面运行检测并生成图片。")
    st.stop()

cols = st.columns(4)
selected_index = st.session_state.get("selected_image_index", None)
for i, p in enumerate(st.session_state.top3_image_paths):
    with cols[i]:
        img = Image.open(p)
        st.image(img, use_container_width=True, caption=f"图片 {i+1}")
        if st.button(f"选择图片 {i+1}", key=i):
            st.session_state.selected_image_index = i
            selected_index = i

with cols[3]:
    if selected_index is None:
        st.info("请选择左侧图片进行处理")
        st.stop()
    path = st.session_state.top3_image_paths[selected_index]
    sel_img = Image.open(path)
    st.image(sel_img, use_container_width=True, caption="已选图片")

    # 保存到 session
    buf = io.BytesIO()
    sel_img.save(buf, format="PNG")
    st.session_state.selected_img_bytes = buf.getvalue()
    arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    st.session_state.cv2_selected_img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

# 2️⃣ 初始化模型
with st.spinner("正在初始化模型..."):
    if "unwrap_model" not in st.session_state:
        st.session_state.unwrap_model = load_uvdoc_model("./weights/unwrap_model/best_model.pkl",map_location=torch.device("cpu"))

# 侧边栏配置
with st.sidebar:
    st.header("模型配置")
    sel_unwrap = st.selectbox(
        "几何校正模型",
        os.listdir("./weights/unwrap_model"),
        index=0
    )
    sel_yolo = st.selectbox(
        "分类检测模型",
        os.listdir("./weights/cls_model"),
        index=0
    )
    # 切换校正模型
    if st.session_state.get("current_unwrap") != sel_unwrap:
        st.session_state.unwrap_model = load_uvdoc_model(
            os.path.join("./weights/unwrap_model", sel_unwrap)
        )
        st.session_state.current_unwrap = sel_unwrap
    ocr_prompt = st.text_area("OCR 提示词", OCR_PROMPT_DEFAULT)

st.markdown("---")

# 3️⃣ 原图、检测、校正布局
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("原始图片")
    st.image(st.session_state.selected_img_bytes, use_container_width=True)

with col2:
    st.subheader("文档检测")
    if st.button("运行检测"):
        fd, tmp_path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        cv2.imwrite(tmp_path, st.session_state.cv2_selected_img)
        ymodel = load_yolo_model(os.path.join("./weights/cls_model", sel_yolo))
        results = ymodel(tmp_path)
        os.remove(tmp_path)
        ann = plot_results(st.session_state.cv2_selected_img, results)
        st.session_state.annotated_img = ann
    if "annotated_img" in st.session_state:
        st.image(
            st.session_state.annotated_img,
            channels="BGR",
            use_container_width=True,
            caption="检测结果"
        )

with col3:
    st.subheader("校正图像")
    if st.button("运行校正"):
        fd2, tmp_path2 = tempfile.mkstemp(suffix=".png")
        os.close(fd2)
        cv2.imwrite(tmp_path2, st.session_state.cv2_selected_img)
        proc_path = unwarp_img(
            os.path.join("./weights/unwrap_model", sel_unwrap),
            tmp_path2,
            IMG_SIZE,
            st.session_state.unwrap_model
        )
        os.remove(tmp_path2)
        st.session_state.processed_img = proc_path
    if "processed_img" in st.session_state:
        st.image(
            st.session_state.processed_img,
            use_container_width=True,
            caption="校正后图像"
        )


#ocr部分，支持在线cpu
st.subheader("OCR 可视化结果")
if st.button("运行 OCR 可视化"):
    # 确定输入图像路径
    input_path = st.session_state.get("processed_img", None)
    if input_path is None:
        # 临时保存原图
        fd, tmp = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        cv2.imwrite(tmp, st.session_state.cv2_selected_img)
        input_path = tmp

    # 调用后端函数
    vis_img, texts, boxes = run_cnocr_visualization(
        img_fp         = input_path,
        rec_model_fp   = "./weights/ocr_model/ocr_densenet.onnx",
        det_model_fp   = "./weights/ocr_model/ch_PP-OCRv4_det_infer.onnx",
        font_path      = "./fonts/simhei.ttf",
        font_size      = 20
    )

    # 若写了临时文件，删除之
    if 'tmp' in locals():
        os.remove(tmp)

    # 三列展示：可视化图 / 文字 / 坐标
    c1, c2, c3 = st.columns(3)
    # 转 RGB
    rgb = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
    with c1:
        st.image(rgb, use_container_width=True, caption="OCR 可视化")
    with c2:
        st.markdown("**识别文字**")
        for i, t in enumerate(texts,1):
            st.write(f"{i}. {t}")
    with c3:
        st.markdown("**文字坐标**")
        for i, b in enumerate(boxes,1):
            x0,y0 = b[0]; x2,y2 = b[2]
            st.write(f"{i}. ({x0},{y0})→({x2},{y2})")
# # 视觉模型部分，如果是在线需要接入api
# st.markdown("---")
# st.subheader("视觉模型")
# if st.button("运行"):
#     if "processed_img" in st.session_state:
#         ocr_res = perform_ocr(ocr_prompt, st.session_state.processed_img)
#         st.session_state.ocr_result = ocr_res
# if "ocr_result" in st.session_state:
#     st.text_area("模型返回结果", st.session_state.ocr_result, height=200)

st.markdown(
    "<p style='text-align: center; color: gray;'>© 2025 计算机视觉课程设计 · Authored by 万励为</p>",
    unsafe_allow_html=True
)