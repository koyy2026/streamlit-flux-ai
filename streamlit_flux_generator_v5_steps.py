import streamlit as st
import requests
from PIL import Image
from io import BytesIO
from openai import OpenAI
import zipfile
import io
import base64
import json
from datetime import datetime
import os

st.set_page_config(page_title="Flux AI 圖像生成器 (v5)", layout="wide")
st.title("🎨 Flux AI 圖像生成器 (v5) - 步數控制版")

# Initialize session state for image history
if 'image_history' not in st.session_state:
    st.session_state.image_history = []

# API 配置區
st.sidebar.header("API 配置")
api_key = st.sidebar.text_input("API Key", type="password")
base_url = st.sidebar.text_input("Base URL", "https://api.navy/v1")

# 功能模式選擇
st.sidebar.header("功能選擇")
generation_mode = st.sidebar.radio(
    "選擇生成模式",
    ["文生圖 (Text-to-Image)", "圖生圖 (Image-to-Image)"]
)

# 模型選擇
models = [
    "flux.1-schnell",
    "flux.1.1-por", 
    "flux.latest",
    "flux.1-krea-dev",
    "flux.1-kontext-pro",
    "flux.1-kontext-max"
]

model = st.sidebar.selectbox("選擇模型", models, index=0)

# 根據模型動態配置 style、quality 和 steps (英文版本)
styles_dict = {
    "flux.1-schnell": ["vivid", "natural", "fantasy", "anime", "monochrome", "watercolor", "sketch", "oil_painting"],
    "flux.1.1-por": ["cinematic", "photographic", "anime", "monochrome", "watercolor", "sketch", "oil_painting"],
    "flux.latest": ["modern", "retro", "anime", "monochrome", "watercolor", "sketch", "oil_painting"],
    "flux.1-krea-dev": ["style1", "style2", "style3", "anime", "monochrome", "watercolor", "sketch", "oil_painting"],
    "flux.1-kontext-pro": ["styleA", "styleB", "anime", "monochrome", "watercolor", "sketch", "oil_painting"],
    "flux.1-kontext-max": ["styleX", "styleY", "anime", "monochrome", "watercolor", "sketch", "oil_painting"]
}

qualities_dict = {
    "flux.1-schnell": ["standard", "hd"],
    "flux.1.1-por": ["standard", "hd", "ultra_hd"],
    "flux.latest": ["standard", "hd", "ultra_hd"],
    "flux.1-krea-dev": ["standard", "hd"],
    "flux.1-kontext-pro": ["standard", "hd", "ultra_hd"],
    "flux.1-kontext-max": ["standard", "hd", "ultra_hd"]
}

# 模型對應的默認和推薦步數範圍
steps_dict = {
    "flux.1-schnell": {"default": 4, "min": 1, "max": 8, "recommended": [1, 2, 4, 8]},
    "flux.1.1-por": {"default": 20, "min": 10, "max": 50, "recommended": [10, 15, 20, 25, 30]},
    "flux.latest": {"default": 20, "min": 10, "max": 50, "recommended": [15, 20, 25, 30, 40]},
    "flux.1-krea-dev": {"default": 25, "min": 10, "max": 50, "recommended": [15, 20, 25, 30]},
    "flux.1-kontext-pro": {"default": 28, "min": 15, "max": 50, "recommended": [20, 25, 28, 35, 40]},
    "flux.1-kontext-max": {"default": 30, "min": 20, "max": 50, "recommended": [25, 28, 30, 35, 40]}
}

# 風格和品質選擇
style = st.sidebar.selectbox("風格", styles_dict[model], index=0)
quality = st.sidebar.selectbox("品質", qualities_dict[model], index=0)

# 生成步數控制
st.sidebar.header("生成參數")
steps_config = steps_dict[model]

# 步數選擇方式
step_mode = st.sidebar.radio(
    "步數選擇方式",
    ["推薦值", "自定義"],
    help="推薦值：使用預設的最佳步數；自定義：手動調整步數"
)

if step_mode == "推薦值":
    recommended_steps = steps_config["recommended"]
    step_labels = [f"{s} 步 {'(默認)' if s == steps_config['default'] else '(快速)' if s < steps_config['default'] else '(高品質)'}" 
                  for s in recommended_steps]
    selected_step_label = st.sidebar.selectbox("選擇步數", step_labels)
    steps = recommended_steps[step_labels.index(selected_step_label)]
else:
    steps = st.sidebar.slider(
        "生成步數", 
        min_value=steps_config["min"], 
        max_value=steps_config["max"], 
        value=steps_config["default"],
        step=1,
        help=f"步數越高品質越好但生成時間越長。{model} 推薦範圍：{steps_config['min']}-{steps_config['max']}"
    )

# 顯示步數信息
steps_info = ""
if steps <= steps_config["default"] // 2:
    steps_info = "⚡ 超快速模式"
elif steps < steps_config["default"]:
    steps_info = "🚀 快速模式"
elif steps == steps_config["default"]:
    steps_info = "⚖️ 平衡模式 (推薦)"
elif steps <= steps_config["default"] * 1.5:
    steps_info = "🎨 高品質模式"
else:
    steps_info = "💎 極致品質模式"

st.sidebar.info(f"當前設定：{steps} 步 - {steps_info}")

# 圖像參數
size = st.sidebar.selectbox(
    "圖像尺寸", 
    ["1024x1024", "1024x1792", "1792x1024", "1024x1536", "1536x1024"],
    index=0
)
n_images = st.sidebar.slider("生成數量", 1, 5, 1)

# 高級參數
with st.sidebar.expander("🔧 高級參數"):
    guidance_scale = st.slider(
        "引導強度 (Guidance Scale)", 
        1.0, 20.0, 7.5, 0.5,
        help="控制模型對提示詞的遵循程度。數值越高越嚴格遵循提示詞"
    )

    seed = st.number_input(
        "隨機種子 (可選)", 
        min_value=-1, max_value=2147483647, value=-1,
        help="設定固定種子可重現相同結果。-1 為隨機種子"
    )

# 圖生圖專用參數
if generation_mode == "圖生圖 (Image-to-Image)":
    st.sidebar.header("圖生圖參數")

    # 圖片上傳選項
    upload_option = st.sidebar.radio(
        "選擇圖片來源",
        ["上傳新圖片", "從歷史記錄選擇"]
    )

    source_image = None

    if upload_option == "上傳新圖片":
        uploaded_file = st.sidebar.file_uploader(
            "上傳參考圖片", 
            type=["png", "jpg", "jpeg"],
            help="支援 PNG, JPG, JPEG 格式"
        )
        if uploaded_file:
            source_image = Image.open(uploaded_file)

    elif upload_option == "從歷史記錄選擇" and st.session_state.image_history:
        history_options = [f"圖片 {i+1} - {item['timestamp']}" for i, item in enumerate(st.session_state.image_history)]
        selected_history = st.sidebar.selectbox("選擇歷史圖片", history_options)
        if selected_history:
            history_index = int(selected_history.split()[1]) - 1
            # Convert base64 back to image
            image_data = base64.b64decode(st.session_state.image_history[history_index]['image_data'])
            source_image = Image.open(BytesIO(image_data))

    # 圖生圖強度控制
    strength = st.sidebar.slider(
        "變化強度", 
        0.1, 1.0, 0.7, 0.1,
        help="數值越高，生成的圖片與原圖差異越大"
    )

# 主要生成區域
st.header("圖像生成")

# 提示詞輸入
if generation_mode == "文生圖 (Text-to-Image)":
    prompt = st.text_area("描述您想要的圖像", height=100, placeholder="輸入詳細的圖像描述...")
else:
    prompt = st.text_area("圖像修改描述", height=100, placeholder="描述您希望如何修改圖片...")

# 負面提示詞
negative_prompt = st.text_area("負面提示詞 (可選)", height=60, placeholder="描述不希望出現的元素...")

# 預計生成時間估算
estimated_time = steps * 0.5 * n_images  # 粗略估算：每步0.5秒
if estimated_time < 60:
    time_estimate = f"預計生成時間：約 {estimated_time:.0f} 秒"
else:
    time_estimate = f"預計生成時間：約 {estimated_time/60:.1f} 分鐘"

st.info(f"📊 生成設定：{steps} 步 | {n_images} 張圖片 | {time_estimate}")

# 生成按鈕
if st.button("🎨 生成圖像", type="primary"):
    if not api_key:
        st.error("請輸入 API Key")
    elif not prompt:
        st.error("請輸入提示詞")
    elif generation_mode == "圖生圖 (Image-to-Image)" and source_image is None:
        st.error("請上傳或選擇參考圖片")
    else:
        with st.spinner(f"正在生成圖像... (預計 {estimated_time:.0f} 秒)"):
            try:
                client = OpenAI(api_key=api_key, base_url=base_url)

                # 準備請求參數
                request_params = {
                    "model": model,
                    "prompt": prompt,
                    "n": n_images,
                    "size": size,
                    "quality": quality,
                    "style": style,
                    "steps": steps,
                    "guidance_scale": guidance_scale
                }

                # 添加負面提示詞
                if negative_prompt:
                    request_params["negative_prompt"] = negative_prompt

                # 添加種子（如果不是-1）
                if seed != -1:
                    request_params["seed"] = seed

                # 圖生圖模式的特殊處理
                if generation_mode == "圖生圖 (Image-to-Image)":
                    # Convert image to base64
                    buffered = BytesIO()
                    source_image.save(buffered, format="PNG")
                    image_base64 = base64.b64encode(buffered.getvalue()).decode()

                    request_params["image"] = image_base64
                    request_params["strength"] = strength

                # 調用 API
                response = client.images.generate(**request_params)

                # 處理生成的圖像
                images = []
                for i, image_data in enumerate(response.data):
                    if hasattr(image_data, 'url'):
                        # 從 URL 下載圖像
                        img_response = requests.get(image_data.url)
                        img = Image.open(BytesIO(img_response.content))
                    elif hasattr(image_data, 'b64_json'):
                        # 從 base64 解碼圖像
                        img_data = base64.b64decode(image_data.b64_json)
                        img = Image.open(BytesIO(img_data))
                    else:
                        continue

                    images.append(img)

                    # 保存到歷史記錄
                    buffered = BytesIO()
                    img.save(buffered, format="PNG")
                    img_base64 = base64.b64encode(buffered.getvalue()).decode()

                    history_item = {
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "prompt": prompt,
                        "model": model,
                        "style": style,
                        "quality": quality,
                        "size": size,
                        "steps": steps,
                        "guidance_scale": guidance_scale,
                        "mode": generation_mode,
                        "image_data": img_base64
                    }

                    if negative_prompt:
                        history_item["negative_prompt"] = negative_prompt

                    if seed != -1:
                        history_item["seed"] = seed

                    if generation_mode == "圖生圖 (Image-to-Image)":
                        history_item["strength"] = strength

                    st.session_state.image_history.append(history_item)

                # 顯示生成的圖像
                if images:
                    st.success(f"🎉 成功生成 {len(images)} 張圖像! (用時約 {estimated_time:.0f} 秒)")

                    # 使用列布局顯示圖像
                    cols = st.columns(min(3, len(images)))

                    for i, img in enumerate(images):
                        with cols[i % 3]:
                            st.image(img, caption=f"生成圖像 {i+1} - {steps} 步", use_container_width=True)

                    # 批量下載功能
                    if len(images) > 1:
                        zip_buffer = BytesIO()
                        with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
                            for i, img in enumerate(images):
                                img_buffer = BytesIO()
                                img.save(img_buffer, format='PNG')
                                zip_file.writestr(f'generated_image_{i+1}_{steps}steps.png', img_buffer.getvalue())

                        st.download_button(
                            label="📦 下載所有圖片 (ZIP)",
                            data=zip_buffer.getvalue(),
                            file_name=f"flux_images_{steps}steps_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                            mime="application/zip"
                        )
                else:
                    st.error("圖像生成失敗")

            except Exception as e:
                st.error(f"生成失敗: {str(e)}")

# 圖片歷史記錄區域
if st.session_state.image_history:
    st.header("📚 圖片歷史記錄")

    # 歷史記錄控制
    col1, col2 = st.columns([3, 1])
    with col1:
        st.write(f"共有 {len(st.session_state.image_history)} 張歷史圖片")
    with col2:
        if st.button("🗑️ 清空歷史"):
            st.session_state.image_history = []
            st.rerun()

    # 顯示最近的圖片
    recent_count = st.slider("顯示最近圖片數量", 1, min(20, len(st.session_state.image_history)), 6)

    recent_images = st.session_state.image_history[-recent_count:][::-1]  # 最新的在前

    # 分組顯示歷史圖片
    for i in range(0, len(recent_images), 3):
        cols = st.columns(3)
        for j in range(3):
            if i + j < len(recent_images):
                item = recent_images[i + j]
                with cols[j]:
                    # 顯示圖片
                    image_data = base64.b64decode(item['image_data'])
                    img = Image.open(BytesIO(image_data))
                    st.image(img, caption=f"{item['steps']} 步生成", use_container_width=True)

                    # 顯示詳細信息
                    with st.expander(f"詳細信息 - {item['timestamp']}"):
                        st.write(f"**提示詞:** {item['prompt']}")
                        st.write(f"**模型:** {item['model']}")
                        st.write(f"**風格:** {item['style']}")
                        st.write(f"**品質:** {item['quality']}")
                        st.write(f"**尺寸:** {item['size']}")
                        st.write(f"**步數:** {item['steps']} (關鍵參數)")
                        st.write(f"**引導強度:** {item['guidance_scale']}")
                        st.write(f"**模式:** {item['mode']}")
                        if 'negative_prompt' in item:
                            st.write(f"**負面提示詞:** {item['negative_prompt']}")
                        if 'seed' in item:
                            st.write(f"**隨機種子:** {item['seed']}")
                        if 'strength' in item:
                            st.write(f"**變化強度:** {item['strength']}")

                    # 單張下載按鈕
                    st.download_button(
                        label="💾 下載",
                        data=base64.b64decode(item['image_data']),
                        file_name=f"flux_image_{item['steps']}steps_{item['timestamp'].replace(':', '-').replace(' ', '_')}.png",
                        mime="image/png",
                        key=f"download_{len(st.session_state.image_history)-recent_count+i+j}"
                    )

# 使用說明
with st.expander("📖 使用說明"):
    st.markdown("""
    ### 功能特色
    - **精確步數控制**: 支援不同模型的最佳步數範圍
    - **智能推薦**: 提供預設和推薦步數選項
    - **時間估算**: 實時顯示預計生成時間
    - **高級參數**: 支援引導強度和隨機種子控制
    - **完整記錄**: 保存所有生成參數包括步數

    ### 步數選擇指南
    - **快速模式 (低步數)**: 適合快速預覽和概念驗證
    - **平衡模式 (默認步數)**: 品質與速度的最佳平衡
    - **高品質模式 (高步數)**: 適合最終作品和精細調整

    ### 模型步數建議
    - **flux.1-schnell**: 1-8 步，專為快速生成優化
    - **flux.1.1-por**: 10-50 步，適合高品質攝影風格
    - **flux.latest**: 10-50 步，現代風格的平衡選擇
    - **專業版本**: 15-50 步，提供最佳品質輸出

    ### 高級功能
    - **引導強度**: 控制對提示詞的遵循程度 (1-20)
    - **隨機種子**: 設定固定值可重現相同結果
    - **批量生成**: 同時生成多張不同變化的圖片
    """)
