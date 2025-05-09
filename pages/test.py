import streamlit as st
import random
import string
from PIL import Image, ImageDraw, ImageFont
import io

# === 初始化 session_state 变量 ===
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "captcha_text" not in st.session_state:
    st.session_state.captcha_text = ""

# === 验证码生成函数 ===
def generate_captcha():
    text = ''.join(random.choices(string.ascii_uppercase + string.digits, k=5))
    image = Image.new('RGB', (120, 40), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 24)  # 如果在 Linux 上出错可注释这行
    except:
        font = ImageFont.load_default()
    draw.text((10, 5), text, font=font, fill=(0, 0, 0))
    return text, image

# === 登录表单 ===
if not st.session_state.logged_in:
    st.title("🔐 登录界面（含验证码）")

    username = st.text_input("用户名")
    password = st.text_input("密码", type="password")

    # 生成验证码图片
    if st.button("刷新验证码") or st.session_state.captcha_text == "":
        st.session_state.captcha_text, captcha_image = generate_captcha()
    else:
        _, captcha_image = generate_captcha()
        captcha_image = captcha_image.copy()
        draw = ImageDraw.Draw(captcha_image)
        draw.text((10, 5), st.session_state.captcha_text, fill=(0, 0, 0))

    # 显示验证码图像
    buf = io.BytesIO()
    captcha_image.save(buf, format="PNG")
    st.image(buf.getvalue(), caption="请输入上图中的验证码")

    # 用户输入验证码
    captcha_input = st.text_input("验证码")

    # 登录按钮
    if st.button("登录"):
        # 简单用户名密码验证（可改为查数据库）
        if username == "admin" and password == "123456":
            if captcha_input.upper() == st.session_state.captcha_text:
                st.success("✅ 登录成功！")
                st.session_state.logged_in = True
            else:
                st.error("❌ 验证码错误！")
        else:
            st.error("❌ 用户名或密码错误！")

# === 登录成功后的页面 ===
else:
    st.title("🎉 欢迎来到主页")
    st.write("你已成功登录。")
    if st.button("退出登录"):
        st.session_state.logged_in = False
        st.session_state.captcha_text = ""
