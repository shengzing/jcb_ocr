"""
Streamlit交互界面 - OCR模型混合使用

功能:
1. 选择布局识别模型(4个模型可选)
2. 选择内容识别模型(4个模型可选)
3. 上传文件(支持图片和PDF)
4. 文件解析(调用配置的模型进行解析,实时显示日志)
5. 结果预览(显示JSON结果和可视化)
"""

import sys
import os
from pathlib import Path
import json
import tempfile
import io
from datetime import datetime

# 添加项目根目录到路径
_current_dir = Path(__file__).resolve()
app_root = _current_dir.parent.parent  # jcb_ocr 根目录
project_root = app_root.parent         # 保持与既有逻辑兼容
sys.path.insert(0, str(project_root))
LOGO_PATH = app_root / "images" / "logo.png"

import streamlit as st
from PIL import Image
import fitz  # PyMuPDF for PDF handling
from dotenv import load_dotenv

# 加载 .env 文件
env_path = project_root / "vlm_client" / ".env"
load_dotenv(env_path)

# 导入OCR客户端
from vlm_client.local_vl_utils.qwen_client import QwenClient
from vlm_client.local_vl_utils.deepseek_client import DeepSeekClient
from vlm_client.local_vl_utils.hunyuan_client import HunyuanClient
from vlm_client.local_vl_utils.mineru_client import MinerUClient
from vlm_client.local_vl_utils.format_converter import (
    blocks_to_standard_json,
    blocks_to_json
)

# spans PDF生成功能将在需要时动态导入


# ==================== 配置 ====================

st.set_page_config(
    page_title="JCB-OCR混合模型解析系统",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
    <style>
    /* 全局字体大小调整 */
    html, body, [class*="css"] {
        font-size: 14px;
    }

    /* 标题字体优化 */
    h1 {
        font-size: 1.8rem !important;
        font-weight: 600 !important;
        margin-bottom: 1rem !important;
    }

    h2 {
        font-size: 1.4rem !important;
        font-weight: 600 !important;
        margin-top: 1.5rem !important;
        margin-bottom: 0.8rem !important;
    }

    h3 {
        font-size: 1.2rem !important;
        font-weight: 600 !important;
        margin-bottom: 0.6rem !important;
    }

    /* 侧边栏样式优化 */
    .css-1d391kg, [data-testid="stSidebar"] {
        background-color: #f8f9fa;
    }

    .css-1d391kg p, [data-testid="stSidebar"] p {
        font-size: 0.85rem;
    }

    /* 按钮样式优化 */
    .stButton button {
        font-size: 0.9rem !important;
        padding: 0.4rem 1rem !important;
        border-radius: 6px !important;
    }

    /* 文本区域字体 */
    .stTextArea textarea {
        font-size: 0.8rem !important;
        font-family: 'Monaco', 'Menlo', 'Consolas', monospace !important;
    }

    /* 选择框样式 */
    .stSelectbox label, .stFileUploader label {
        font-size: 0.9rem !important;
        font-weight: 500 !important;
    }

    /* 信息框样式 */
    .stInfo, .stSuccess, .stWarning, .stError {
        font-size: 0.85rem !important;
        padding: 0.6rem !important;
    }

    /* Metric样式优化 */
    [data-testid="stMetricValue"] {
        font-size: 1.5rem !important;
    }

    [data-testid="stMetricLabel"] {
        font-size: 0.85rem !important;
    }

    /* Expander样式 */
    .streamlit-expanderHeader {
        font-size: 0.9rem !important;
    }

    /* Caption文字 */
    .css-16idsys p, small, .stCaption {
        font-size: 0.75rem !important;
        color: #6c757d !important;
    }

    /* 表格样式 */
    .dataframe {
        font-size: 0.85rem !important;
    }

    /* 下载按钮优化 */
    .stDownloadButton button {
        font-size: 0.85rem !important;
    }

    /* 进度条容器 */
    .stProgress > div > div {
        height: 6px !important;
    }

    /* JSON显示区域 */
    .stJson {
        font-size: 0.8rem !important;
    }

    /* 优化间距 */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
    }

    /* 文件上传器样式 */
    [data-testid="stFileUploader"] section {
        padding: 1rem !important;
    }

    /* 标签页样式 */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
    }

    .stTabs [data-baseweb="tab"] {
        font-size: 0.9rem !important;
        padding: 0.5rem 1rem !important;
    }
    </style>
""", unsafe_allow_html=True)

# 模型配置
MODEL_CONFIGS = {
    "Qwen-VL": {
        "class": QwenClient,
        "description": "Qwen-VL-7B模型,支持JSON/Legacy格式输出",
        "env_prefix": "QWEN"
    },
    "DeepSeek-VL": {
        "class": DeepSeekClient,
        "description": "DeepSeek-VL2模型,支持Grounding格式",
        "env_prefix": "DEEPSEEK"
    },
    "Hunyuan-VL": {
        "class": HunyuanClient,
        "description": "Hunyuan-VL模型,支持XML格式输出",
        "env_prefix": "HUNYUAN"
    },
    "MinerU": {
        "class": MinerUClient,
        "description": "MinerU模型,专注于文档解析",
        "env_prefix": "MINERU"
    }
}


# ==================== 辅助函数 ====================

def init_session_state():
    """初始化session state"""
    if 'logs' not in st.session_state:
        st.session_state.logs = []
    if 'parsed_results' not in st.session_state:
        st.session_state.parsed_results = None
    if 'uploaded_images' not in st.session_state:
        st.session_state.uploaded_images = []
    if 'comparison_results' not in st.session_state:
        # 存储对比结果: {filename: [{model_config, results, timestamp}, ...]}
        st.session_state.comparison_results = {}
    if 'uploaded_filenames' not in st.session_state:
        st.session_state.uploaded_filenames = []


def log_message(message: str, level: str = "INFO"):
    """添加日志消息"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] [{level}] {message}"
    st.session_state.logs.append(log_entry)
    return log_entry


def create_client(model_name: str, server_url: str = None):
    """创建模型客户端,从 .env 读取配置参数"""
    try:
        model_config = MODEL_CONFIGS[model_name]
        env_prefix = model_config["env_prefix"]

        # 从环境变量读取配置
        api_key = os.getenv(f"{env_prefix}_API_KEY")
        model_name_env = os.getenv(f"{env_prefix}_MODEL_NAME")
        server_url_env = os.getenv(f"{env_prefix}_SERVER_URL")

        # 使用传入的 server_url 或环境变量中的值
        final_server_url = server_url if server_url else server_url_env

        # 从环境变量读取通用配置
        # 优先使用模型特定的超时配置,否则使用默认值
        timeout = int(os.getenv(f"{env_prefix}_TIMEOUT") or os.getenv("DEFAULT_TIMEOUT", "600"))
        max_retries = int(os.getenv(f"{env_prefix}_MAX_RETRIES") or os.getenv("DEFAULT_MAX_RETRIES", "3"))

        log_message(f"正在创建{model_name}客户端...")
        log_message(f"  服务地址: {final_server_url}")
        log_message(f"  模型名称: {model_name_env}")
        log_message(f"  超时时间: {timeout}秒")

        # 构建 server_headers (用于传递 API Key)
        server_headers = None
        if api_key:
            server_headers = {"Authorization": f"Bearer {api_key}"}

        client = model_config["class"](
            backend="http-client",
            server_url=final_server_url,
            model_name=model_name_env,
            server_headers=server_headers,
            http_timeout=timeout,
            max_retries=max_retries,
            use_tqdm=False
        )
        log_message(f"成功创建{model_name}客户端", "SUCCESS")
        return client
    except Exception as e:
        log_message(f"创建{model_name}客户端失败: {str(e)}", "ERROR")
        raise


def pdf_to_images(pdf_file) -> list[Image.Image]:
    """将PDF转换为图像列表"""
    log_message(f"开始转换PDF文件,大小: {len(pdf_file.getvalue())} 字节")
    images = []

    try:
        # 使用PyMuPDF打开PDF
        pdf_bytes = pdf_file.getvalue()
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")

        log_message(f"PDF共有 {len(pdf_document)} 页")

        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]

            # 渲染为图像(提高分辨率)
            mat = fitz.Matrix(2.0, 2.0)  # 2x缩放
            pix = page.get_pixmap(matrix=mat)

            # 转换为PIL Image
            img_data = pix.tobytes("png")
            image = Image.open(io.BytesIO(img_data))
            images.append(image)

            log_message(f"  已转换第 {page_num + 1} 页,尺寸: {image.size}")

        pdf_document.close()
        log_message(f"PDF转换完成,共 {len(images)} 张图像", "SUCCESS")
        return images

    except Exception as e:
        log_message(f"PDF转换失败: {str(e)}", "ERROR")
        raise


def process_image(
    image: Image.Image,
    layout_client,
    content_client,
    layout_model_name: str,
    content_model_name: str,
    image_index: int = 0
) -> dict:
    """处理单张图像"""
    log_message(f"=" * 60)
    log_message(f"开始处理第 {image_index + 1} 张图像,尺寸: {image.size}")

    try:
        # 第一步: 布局检测
        log_message(f"步骤1: 使用 {layout_model_name} 进行布局检测...")
        layout_blocks = layout_client.layout_detect(image)
        log_message(f"  检测到 {len(layout_blocks)} 个布局块", "SUCCESS")

        # 第二步: 内容提取
        if layout_model_name == content_model_name:
            # 同一个模型,直接用two_step_extract
            log_message(f"步骤2: 使用 {content_model_name} 进行内容提取(同一模型优化)...")
            log_message(f"  使用two_step_extract一次性完成布局+内容...")
            content_blocks = layout_client.two_step_extract(image)
            log_message(f"  提取完成,共 {len(content_blocks)} 个内容块", "SUCCESS")
        else:
            # 不同模型,需要手动提取
            log_message(f"步骤2: 使用 {content_model_name} 进行内容提取(跨模型)...")
            block_images, prompts, params, indices = content_client.helper.prepare_for_extract(
                image, layout_blocks
            )
            log_message(f"  准备提取 {len(block_images)} 个内容块...")

            if len(block_images) > 0:
                log_message(f"  正在批量请求VLM服务进行内容识别...")
                outputs = content_client.client.batch_predict(
                    block_images, prompts, params, priority=None
                )
                log_message(f"  VLM服务返回 {len(outputs)} 个结果")

                log_message(f"  正在解析并填充内容...")
                for idx, output in zip(indices, outputs):
                    layout_blocks[idx].content = content_client._extract_text(output)
                log_message(f"  内容填充完成")

            # 后处理
            log_message(f"  正在进行后处理...")
            content_blocks = content_client.helper.post_process(layout_blocks)
            log_message(f"  提取完成,共 {len(content_blocks)} 个内容块", "SUCCESS")

        # 转换为统一JSON格式
        log_message(f"步骤3: 转换为标准JSON格式...")
        model_type = f"{layout_model_name}_layout_{content_model_name}_content"
        result = blocks_to_standard_json(
            content_blocks,
            model_type=model_type,
            include_metadata=True
        )
        log_message(f"  JSON转换完成,包含 {len(result.get('blocks', []))} 个块")

        log_message(f"第 {image_index + 1} 张图像处理完成 ✓", "SUCCESS")
        return result

    except Exception as e:
        log_message(f"处理第 {image_index + 1} 张图像失败: {str(e)}", "ERROR")
        import traceback
        log_message(traceback.format_exc(), "ERROR")
        return {"error": str(e), "blocks": [], "metadata": {"error": True}}


def crop_block_from_image(image: Image.Image, bbox: list, padding: int = 5) -> Image.Image:
    """从图像中裁剪出指定bbox的区域

    Args:
        image: 原始图像
        bbox: 归一化坐标 [x1, y1, x2, y2]，范围0-1
        padding: 裁剪时额外的边距(像素)

    Returns:
        裁剪后的图像
    """
    width, height = image.size

    # 转换为像素坐标
    x1 = int(bbox[0] * width)
    y1 = int(bbox[1] * height)
    x2 = int(bbox[2] * width)
    y2 = int(bbox[3] * height)

    # 添加padding，但不超出图像边界
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(width, x2 + padding)
    y2 = min(height, y2 + padding)

    # 裁剪图像
    cropped = image.crop((x1, y1, x2, y2))
    return cropped


def render_blocks_on_image(image: Image.Image, blocks: list) -> Image.Image:
    """在图像上绘制检测到的块"""
    from PIL import ImageDraw, ImageFont

    img_with_boxes = image.copy()
    draw = ImageDraw.Draw(img_with_boxes)

    width, height = image.size

    # 颜色映射
    color_map = {
        "text": "#00FF00",      # 绿色
        "title": "#FF0000",     # 红色
        "table": "#0000FF",     # 蓝色
        "image": "#FFFF00",     # 黄色
        "list": "#FF00FF",      # 品红
        "equation": "#00FFFF",  # 青色
        "handwritten": "#FFA500",  # 橙色
        "seal": "#800080",      # 紫色
    }

    for block in blocks:
        block_type = block.get("type", "unknown")
        bbox = block.get("bbox", [0, 0, 1, 1])

        # 转换为像素坐标
        x1 = int(bbox[0] * width)
        y1 = int(bbox[1] * height)
        x2 = int(bbox[2] * width)
        y2 = int(bbox[3] * height)

        # 选择颜色
        color = color_map.get(block_type, "#808080")

        # 绘制矩形
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        # 绘制标签
        label = f"{block_type}"
        draw.text((x1, y1 - 15), label, fill=color)

    return img_with_boxes


def generate_spans_pdf_from_result(result: dict, original_image: Image.Image) -> bytes:
    """从解析结果生成spans PDF (简化版 - 直接在图像上绘制)

    Args:
        result: OCR解析结果(标准JSON格式)
        original_image: 原始图像

    Returns:
        bytes: 生成的PDF文件字节
    """
    try:
        from PIL import ImageDraw, ImageFont
        from pypdf import PdfReader, PdfWriter, PageObject
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter

        # Span type颜色映射 (RGB 0-255)
        SPAN_TYPE_COLORS = {
            "text": (0, 0, 255),        # 蓝色
            "title": (255, 0, 0),       # 红色
            "table": (255, 255, 0),     # 黄色
            "image": (0, 255, 0),       # 绿色
            "equation": (255, 0, 255),  # 品红
            "handwritten": (255, 140, 0),  # 橙色
            "seal": (220, 20, 60),      # 深红
            "list": (128, 0, 128),      # 紫色
            "default": (128, 128, 128), # 灰色
        }

        # 在图像上绘制spans
        img_with_spans = original_image.copy()
        draw = ImageDraw.Draw(img_with_spans)
        width, height = img_with_spans.size

        blocks = result.get("blocks", [])

        for idx, block in enumerate(blocks):
            block_type = block.get("type", "default")
            bbox = block.get("bbox", [0, 0, 1, 1])

            # 转换为像素坐标
            x1 = int(bbox[0] * width)
            y1 = int(bbox[1] * height)
            x2 = int(bbox[2] * width)
            y2 = int(bbox[3] * height)

            # 选择颜色
            color = SPAN_TYPE_COLORS.get(block_type, SPAN_TYPE_COLORS["default"])

            # 绘制矩形
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

            # 绘制序号
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
            except:
                font = ImageFont.load_default()

            draw.text((x1 + 2, y1 + 2), str(idx), fill=color, font=font)

        # 添加图例
        legend_x = width - 150
        legend_y = 20
        legend_bg = (255, 255, 255, 200)

        # 绘制图例背景
        draw.rectangle([legend_x - 10, legend_y - 10, width - 10, legend_y + len(SPAN_TYPE_COLORS) * 20 + 10],
                      fill=(255, 255, 255), outline=(0, 0, 0))

        draw.text((legend_x, legend_y), "Span Types:", fill=(0, 0, 0), font=font)

        for i, (span_type, color) in enumerate(SPAN_TYPE_COLORS.items()):
            if span_type == "default":
                continue
            y_pos = legend_y + 20 + i * 18

            # 绘制颜色框
            draw.rectangle([legend_x, y_pos, legend_x + 12, y_pos + 12], fill=color, outline=(0, 0, 0))

            # 绘制标签
            draw.text((legend_x + 18, y_pos), span_type, fill=(0, 0, 0), font=font)

        # 将图像转换为PDF
        pdf_buffer = io.BytesIO()
        img_with_spans.save(pdf_buffer, format='PDF', resolution=100.0)
        pdf_bytes = pdf_buffer.getvalue()

        log_message(f"成功生成spans PDF, 共标注 {len(blocks)} 个块", "SUCCESS")
        return pdf_bytes

    except Exception as e:
        log_message(f"生成spans PDF失败: {str(e)}", "ERROR")
        import traceback
        log_message(traceback.format_exc(), "ERROR")
        return None


# ==================== 主界面 ====================

def main():
    """主函数"""
    init_session_state()

    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=120)

    # 标题
    st.title("📄 JCB-OCR混合模型解析系统")
    st.markdown("支持选择不同的布局识别和内容识别模型,灵活组合,实时查看解析日志和结果")

    # 侧边栏 - 模型配置
    st.sidebar.header("⚙️ 模型配置")

    # 对比模式开关
    comparison_mode = st.sidebar.checkbox(
        "🔄 对比模式",
        value=False,
        help="启用后可对同一文件使用不同模型组合进行多次解析并对比结果"
    )

    if comparison_mode:
        st.sidebar.info("💡 对比模式已启用\n\n每次解析结果会保存，可在对比标签页查看")

    # 布局识别模型选择
    st.sidebar.subheader("1️⃣ 布局识别模型")
    layout_model = st.sidebar.selectbox(
        "选择布局识别模型",
        list(MODEL_CONFIGS.keys()),
        index=0,
        help="选择用于检测文档布局的模型,参数从.env文件读取"
    )
    st.sidebar.caption(MODEL_CONFIGS[layout_model]["description"])

    # 显示布局模型当前从.env读取的参数
    layout_env_prefix = MODEL_CONFIGS[layout_model]["env_prefix"]
    layout_model_name = os.getenv(f"{layout_env_prefix}_MODEL_NAME", "未配置")
    layout_server_url = os.getenv(f"{layout_env_prefix}_SERVER_URL", "未配置")

    with st.sidebar.expander("📋 布局模型参数 (来自.env)"):
        st.text(f"模型名称: {layout_model_name}")
        st.text(f"服务地址: {layout_server_url}")
        st.text(f"API Key: {'已配置' if os.getenv(f'{layout_env_prefix}_API_KEY') else '未配置'}")

    # 内容识别模型选择
    st.sidebar.subheader("2️⃣ 内容识别模型")
    content_model = st.sidebar.selectbox(
        "选择内容识别模型",
        list(MODEL_CONFIGS.keys()),
        index=1,
        help="选择用于提取文本内容的模型,参数从.env文件读取"
    )
    st.sidebar.caption(MODEL_CONFIGS[content_model]["description"])

    # 显示内容模型当前从.env读取的参数
    content_env_prefix = MODEL_CONFIGS[content_model]["env_prefix"]
    content_model_name = os.getenv(f"{content_env_prefix}_MODEL_NAME", "未配置")
    content_server_url = os.getenv(f"{content_env_prefix}_SERVER_URL", "未配置")

    with st.sidebar.expander("📋 内容模型参数 (来自.env)"):
        st.text(f"模型名称: {content_model_name}")
        st.text(f"服务地址: {content_server_url}")
        st.text(f"API Key: {'已配置' if os.getenv(f'{content_env_prefix}_API_KEY') else '未配置'}")

    # 显示模型组合
    st.sidebar.markdown("---")
    st.sidebar.info(f"**当前配置:**\n\n"
                   f"🔍 布局: {layout_model}\n\n"
                   f"📝 内容: {content_model}\n\n"
                   f"💡 两个模型解耦，中间格式统一")

    # 主界面 - 三栏布局
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("📁 文件上传")

        # 文件上传
        uploaded_files = st.file_uploader(
            "上传图片或PDF文件",
            type=["png", "jpg", "jpeg", "pdf"],
            accept_multiple_files=True,
            help="支持多个文件上传"
        )

        if uploaded_files:
            st.success(f"已上传 {len(uploaded_files)} 个文件")

            # 处理上传的文件
            all_images = []
            filenames = []
            for file in uploaded_files:
                if file.type == "application/pdf":
                    st.info(f"📑 处理PDF: {file.name}")
                    images = pdf_to_images(file)
                    all_images.extend(images)
                    # PDF的每一页都使用相同的文件名
                    filenames.extend([file.name] * len(images))
                else:
                    st.info(f"🖼️ 处理图片: {file.name}")
                    image = Image.open(file)
                    all_images.append(image)
                    filenames.append(file.name)

            st.session_state.uploaded_images = all_images
            st.session_state.uploaded_filenames = filenames
            st.success(f"共加载 {len(all_images)} 张图像")

            # 显示缩略图
            st.subheader("预览")
            cols = st.columns(min(4, len(all_images)))
            for idx, img in enumerate(all_images[:8]):  # 最多显示8张
                with cols[idx % 4]:
                    st.image(img, caption=f"第{idx+1}张", use_container_width=True)
            if len(all_images) > 8:
                st.caption(f"... 还有 {len(all_images) - 8} 张图像未显示")

    with col2:
        st.header("🚀 开始解析")

        if st.button("▶️ 开始解析", type="primary", disabled=len(st.session_state.uploaded_images) == 0):
            # 清空之前的日志和结果
            st.session_state.logs = []
            st.session_state.parsed_results = None

            with st.spinner("正在初始化模型..."):
                try:
                    # 创建客户端
                    log_message("=" * 60, "INFO")
                    log_message("开始初始化OCR解析系统", "INFO")
                    log_message("=" * 60, "INFO")

                    # 布局模型和内容模型都从.env读取配置
                    layout_client = create_client(layout_model)
                    content_client = create_client(content_model)

                    # 处理所有图像
                    results = []
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    for idx, image in enumerate(st.session_state.uploaded_images):
                        status_text.text(f"正在处理第 {idx + 1}/{len(st.session_state.uploaded_images)} 张图像...")

                        result = process_image(
                            image,
                            layout_client,
                            content_client,
                            layout_model,
                            content_model,
                            idx
                        )
                        results.append(result)

                        progress = (idx + 1) / len(st.session_state.uploaded_images)
                        progress_bar.progress(progress)

                    # 保存结果
                    st.session_state.parsed_results = results

                    # 如果是对比模式，保存到对比结果中
                    if comparison_mode:
                        model_config_key = f"{layout_model} + {content_model}"
                        for idx, result in enumerate(results):
                            filename = st.session_state.uploaded_filenames[idx]
                            if filename not in st.session_state.comparison_results:
                                st.session_state.comparison_results[filename] = []

                            # 检查是否已存在相同配置的结果
                            existing = [r for r in st.session_state.comparison_results[filename]
                                       if r['model_config'] == model_config_key]
                            if existing:
                                # 更新已有结果
                                existing[0]['result'] = result
                                existing[0]['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                                log_message(f"  更新文件 [{filename}] 的 [{model_config_key}] 解析结果")
                            else:
                                # 添加新结果
                                st.session_state.comparison_results[filename].append({
                                    'model_config': model_config_key,
                                    'layout_model': layout_model,
                                    'content_model': content_model,
                                    'result': result,
                                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                                })
                                log_message(f"  保存文件 [{filename}] 的 [{model_config_key}] 解析结果到对比列表")

                        log_message(f"对比模式: 已保存 {len(results)} 个文件的解析结果", "SUCCESS")

                    status_text.empty()
                    progress_bar.empty()

                    log_message("=" * 60, "INFO")
                    log_message("所有图像处理完成!", "SUCCESS")
                    log_message("=" * 60, "INFO")

                    st.success(f"✅ 成功解析 {len(results)} 张图像!")
                    if comparison_mode:
                        st.info(f"💡 对比模式: 结果已保存，切换到\"对比分析\"标签页查看")
                    st.balloons()

                except Exception as e:
                    log_message(f"解析过程出错: {str(e)}", "ERROR")
                    st.error(f"❌ 解析失败: {str(e)}")

    # 日志显示区域
    st.header("📋 解析日志")
    log_container = st.container()

    with log_container:
        if st.session_state.logs:
            # 显示最近的日志
            log_text = "\n".join(st.session_state.logs[-100:])  # 最多显示最近100条
            st.text_area(
                "实时日志",
                value=log_text,
                height=300,
                disabled=True,
                label_visibility="collapsed"
            )

            # 下载日志按钮
            log_download = "\n".join(st.session_state.logs)
            st.download_button(
                "💾 下载完整日志",
                data=log_download,
                file_name=f"ocr_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
        else:
            st.info("暂无日志,请上传文件并开始解析")

    # 结果显示区域
    st.header("📊 解析结果")

    # 根据是否有对比结果决定标签页
    if st.session_state.comparison_results and comparison_mode:
        tabs = st.tabs(["JSON结果", "可视化预览", "统计信息", "🔄 对比分析", "📊 可视化对比"])
        has_comparison_tab = True
        has_visual_comparison_tab = True
    else:
        tabs = st.tabs(["JSON结果", "可视化预览", "统计信息"])
        has_comparison_tab = False
        has_visual_comparison_tab = False

    if st.session_state.parsed_results:

        with tabs[0]:
            st.subheader("JSON格式结果")

            # 选择要查看的图像
            if len(st.session_state.parsed_results) > 1:
                selected_idx = st.selectbox(
                    "选择图像",
                    range(len(st.session_state.parsed_results)),
                    format_func=lambda x: f"第 {x + 1} 张图像"
                )
            else:
                selected_idx = 0

            result = st.session_state.parsed_results[selected_idx]

            # 显示JSON
            st.json(result)

            # 下载JSON
            json_str = json.dumps(result, ensure_ascii=False, indent=2)
            st.download_button(
                "💾 下载JSON结果",
                data=json_str,
                file_name=f"ocr_result_{selected_idx+1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

        with tabs[1]:
            st.subheader("可视化预览")

            # 选择要查看的图像
            if len(st.session_state.parsed_results) > 1:
                vis_idx = st.selectbox(
                    "选择图像进行可视化",
                    range(len(st.session_state.parsed_results)),
                    format_func=lambda x: f"第 {x + 1} 张图像",
                    key="vis_select"
                )
            else:
                vis_idx = 0

            result = st.session_state.parsed_results[vis_idx]
            image = st.session_state.uploaded_images[vis_idx]

            # 绘制检测框
            try:
                img_with_boxes = render_blocks_on_image(image, result.get("blocks", []))
                st.image(img_with_boxes, caption=f"检测结果 - 第{vis_idx+1}张", use_container_width=True)
            except Exception as e:
                st.error(f"可视化失败: {str(e)}")
                st.image(image, caption=f"原图 - 第{vis_idx+1}张", use_container_width=True)

            # 显示检测到的块列表
            st.subheader("检测到的内容块")
            blocks = result.get("blocks", [])

            # 需要显示截图的类型
            visual_types = {'image', 'seal', 'table', 'handwritten'}

            for i, block in enumerate(blocks):
                block_type = block.get('type', 'unknown')

                with st.expander(f"块 {i+1}: {block_type}"):
                    # 基本信息
                    col1, col2 = st.columns([2, 1])

                    with col1:
                        st.write(f"**类型:** {block.get('type')}")
                        st.write(f"**坐标:** {block.get('bbox')}")
                        st.write(f"**角度:** {block.get('angle')}")
                        if block.get('content'):
                            st.write(f"**内容:**")
                            st.text(block.get('content')[:500])  # 限制显示长度

                    with col2:
                        # 如果是需要显示截图的类型，显示裁剪后的图像
                        if block_type in visual_types:
                            bbox = block.get('bbox')
                            if bbox and len(bbox) == 4:
                                try:
                                    cropped_img = crop_block_from_image(image, bbox, padding=10)
                                    st.image(cropped_img, caption=f"{block_type}截图", use_container_width=True)
                                except Exception as e:
                                    st.error(f"截图失败: {str(e)}")

        with tabs[2]:
            st.subheader("统计信息")

            # 汇总所有结果的统计
            total_blocks = sum(len(r.get("blocks", [])) for r in st.session_state.parsed_results)

            # 统计各类型块的数量
            type_counts = {}
            for result in st.session_state.parsed_results:
                for block in result.get("blocks", []):
                    block_type = block.get("type", "unknown")
                    type_counts[block_type] = type_counts.get(block_type, 0) + 1

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("总图像数", len(st.session_state.parsed_results))
            with col2:
                st.metric("总内容块数", total_blocks)
            with col3:
                avg_blocks = total_blocks / len(st.session_state.parsed_results) if st.session_state.parsed_results else 0
                st.metric("平均块数/图", f"{avg_blocks:.1f}")

            # 块类型分布
            st.subheader("内容块类型分布")
            if type_counts:
                import pandas as pd
                df = pd.DataFrame(list(type_counts.items()), columns=["类型", "数量"])
                df = df.sort_values("数量", ascending=False)
                st.bar_chart(df.set_index("类型"))
                st.dataframe(df, use_container_width=True)

        # 对比分析标签页
        if has_comparison_tab:
            with tabs[3]:
                st.subheader("🔄 对比分析")

                if not st.session_state.comparison_results:
                    st.info("暂无对比数据，请在对比模式下进行解析")
                else:
                    # 选择要对比的文件
                    filenames = list(st.session_state.comparison_results.keys())
                    if len(filenames) > 1:
                        selected_file = st.selectbox(
                            "选择文件进行对比",
                            filenames,
                            help="选择一个文件查看不同模型配置的解析结果对比"
                        )
                    else:
                        selected_file = filenames[0]
                        st.info(f"当前文件: {selected_file}")

                    comparisons = st.session_state.comparison_results[selected_file]

                    if len(comparisons) < 2:
                        st.warning(f"文件 [{selected_file}] 只有 {len(comparisons)} 个解析结果，至少需要2个才能对比")
                        st.info("💡 请更换模型配置后重新解析，以添加更多对比结果")
                    else:
                        st.success(f"找到 {len(comparisons)} 个不同的模型配置结果")

                        # 显示对比概览
                        st.markdown("### 对比概览")
                        import pandas as pd

                        comparison_data = []
                        for comp in comparisons:
                            result = comp['result']
                            blocks = result.get('blocks', [])
                            type_counts_comp = {}
                            for block in blocks:
                                block_type = block.get('type', 'unknown')
                                type_counts_comp[block_type] = type_counts_comp.get(block_type, 0) + 1

                            comparison_data.append({
                                '模型配置': comp['model_config'],
                                '布局模型': comp['layout_model'],
                                '内容模型': comp['content_model'],
                                '总块数': len(blocks),
                                '解析时间': comp['timestamp'],
                                '块类型': ', '.join([f"{k}({v})" for k, v in sorted(type_counts_comp.items())])
                            })

                        df_comparison = pd.DataFrame(comparison_data)
                        st.dataframe(df_comparison, use_container_width=True)

                        # 并排对比
                        st.markdown("### 详细对比")

                        # 选择两个配置进行并排对比
                        col1, col2 = st.columns(2)

                        with col1:
                            config1_idx = st.selectbox(
                                "选择配置1",
                                range(len(comparisons)),
                                format_func=lambda x: comparisons[x]['model_config'],
                                key="config1"
                            )

                        with col2:
                            config2_idx = st.selectbox(
                                "选择配置2",
                                range(len(comparisons)),
                                format_func=lambda x: comparisons[x]['model_config'],
                                index=min(1, len(comparisons)-1),
                                key="config2"
                            )

                        # 显示并排对比
                        col1, col2 = st.columns(2)

                        with col1:
                            comp1 = comparisons[config1_idx]
                            st.markdown(f"#### {comp1['model_config']}")
                            st.caption(f"解析时间: {comp1['timestamp']}")

                            result1 = comp1['result']
                            blocks1 = result1.get('blocks', [])

                            st.metric("总块数", len(blocks1))

                            # 显示块类型分布
                            type_counts1 = {}
                            for block in blocks1:
                                block_type = block.get('type', 'unknown')
                                type_counts1[block_type] = type_counts1.get(block_type, 0) + 1

                            st.write("**块类型分布:**")
                            for block_type, count in sorted(type_counts1.items()):
                                st.write(f"- {block_type}: {count}")

                            # 显示JSON
                            with st.expander("查看完整JSON"):
                                st.json(result1)

                        with col2:
                            comp2 = comparisons[config2_idx]
                            st.markdown(f"#### {comp2['model_config']}")
                            st.caption(f"解析时间: {comp2['timestamp']}")

                            result2 = comp2['result']
                            blocks2 = result2.get('blocks', [])

                            st.metric("总块数", len(blocks2))

                            # 显示块类型分布
                            type_counts2 = {}
                            for block in blocks2:
                                block_type = block.get('type', 'unknown')
                                type_counts2[block_type] = type_counts2.get(block_type, 0) + 1

                            st.write("**块类型分布:**")
                            for block_type, count in sorted(type_counts2.items()):
                                st.write(f"- {block_type}: {count}")

                            # 显示JSON
                            with st.expander("查看完整JSON"):
                                st.json(result2)

                        # 差异分析
                        st.markdown("### 差异分析")
                        diff_col1, diff_col2, diff_col3 = st.columns(3)

                        with diff_col1:
                            block_diff = len(blocks1) - len(blocks2)
                            st.metric(
                                "块数差异",
                                f"{abs(block_diff)}",
                                delta=f"{block_diff:+d}" if block_diff != 0 else "相同"
                            )

                        with diff_col2:
                            types1 = set(type_counts1.keys())
                            types2 = set(type_counts2.keys())
                            unique_types1 = types1 - types2
                            unique_types2 = types2 - types1
                            st.metric(
                                "独有类型",
                                f"配置1: {len(unique_types1)}, 配置2: {len(unique_types2)}"
                            )

                        with diff_col3:
                            common_types = types1 & types2
                            st.metric("共同类型", len(common_types))

                        # 清除对比数据按钮
                        if st.button("🗑️ 清除所有对比数据", type="secondary"):
                            st.session_state.comparison_results = {}
                            st.rerun()

        # 可视化对比标签页
        if has_visual_comparison_tab:
            with tabs[4]:
                st.subheader("📊 可视化对比 - Spans PDF")

                if not st.session_state.comparison_results:
                    st.info("暂无对比数据，请在对比模式下进行解析")
                else:
                    # 选择要对比的文件
                    filenames = list(st.session_state.comparison_results.keys())
                    if len(filenames) > 1:
                        selected_file = st.selectbox(
                            "选择文件进行可视化对比",
                            filenames,
                            help="选择一个文件查看不同模型配置的spans PDF对比",
                            key="visual_comp_file"
                        )
                    else:
                        selected_file = filenames[0]
                        st.info(f"当前文件: {selected_file}")

                    comparisons = st.session_state.comparison_results[selected_file]

                    if len(comparisons) < 2:
                        st.warning(f"文件 [{selected_file}] 只有 {len(comparisons)} 个解析结果，至少需要2个才能对比")
                        st.info("💡 请更换模型配置后重新解析，以添加更多对比结果")
                    else:
                        st.success(f"找到 {len(comparisons)} 个不同的模型配置结果")

                        # 选择两个配置进行对比
                        col1, col2 = st.columns(2)

                        with col1:
                            config1_idx = st.selectbox(
                                "选择配置1",
                                range(len(comparisons)),
                                format_func=lambda x: comparisons[x]['model_config'],
                                key="visual_config1"
                            )

                        with col2:
                            config2_idx = st.selectbox(
                                "选择配置2",
                                range(len(comparisons)),
                                format_func=lambda x: comparisons[x]['model_config'],
                                index=min(1, len(comparisons)-1),
                                key="visual_config2"
                            )

                        # 生成spans PDF按钮
                        if st.button("🎨 生成Spans PDF对比", type="primary"):
                            with st.spinner("正在生成spans PDF..."):
                                comp1 = comparisons[config1_idx]
                                comp2 = comparisons[config2_idx]

                                # 获取原始图像
                                file_idx = st.session_state.uploaded_filenames.index(selected_file)
                                original_image = st.session_state.uploaded_images[file_idx]

                                # 生成两个spans PDF
                                log_message(f"正在为配置1生成spans PDF: {comp1['model_config']}")
                                spans_pdf1 = generate_spans_pdf_from_result(comp1['result'], original_image)

                                log_message(f"正在为配置2生成spans PDF: {comp2['model_config']}")
                                spans_pdf2 = generate_spans_pdf_from_result(comp2['result'], original_image)

                                if spans_pdf1 and spans_pdf2:
                                    st.success("✅ Spans PDF生成成功!")

                                    # 将PDF转换为图像用于预览
                                    def pdf_to_preview_image(pdf_bytes):
                                        """将PDF转换为预览图像"""
                                        try:
                                            pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                                            page = pdf_doc[0]  # 只预览第一页

                                            # 渲染为图像
                                            mat = fitz.Matrix(2.0, 2.0)  # 2x缩放以提高清晰度
                                            pix = page.get_pixmap(matrix=mat)

                                            # 转换为PIL Image
                                            img_data = pix.tobytes("png")
                                            preview_image = Image.open(io.BytesIO(img_data))
                                            pdf_doc.close()

                                            return preview_image
                                        except Exception as e:
                                            log_message(f"PDF预览转换失败: {str(e)}", "ERROR")
                                            return None

                                    # 转换PDF为预览图像
                                    preview_img1 = pdf_to_preview_image(spans_pdf1)
                                    preview_img2 = pdf_to_preview_image(spans_pdf2)

                                    # 并排显示两个PDF预览
                                    col1, col2 = st.columns(2)

                                    with col1:
                                        st.markdown(f"#### {comp1['model_config']}")
                                        st.caption(f"解析时间: {comp1['timestamp']}")

                                        # 显示PDF预览
                                        if preview_img1:
                                            st.image(preview_img1, caption="Spans PDF预览", use_container_width=True)
                                        else:
                                            st.warning("⚠️ PDF预览生成失败")

                                        # 提供下载按钮
                                        st.download_button(
                                            "💾 下载完整Spans PDF",
                                            data=spans_pdf1,
                                            file_name=f"{selected_file.rsplit('.', 1)[0]}_{comp1['layout_model']}_{comp1['content_model']}_spans.pdf",
                                            mime="application/pdf",
                                            key="download_pdf1"
                                        )

                                        # 显示统计信息
                                        with st.expander("📊 查看统计信息"):
                                            blocks1 = comp1['result'].get('blocks', [])
                                            st.metric("总块数", len(blocks1))

                                            type_counts1 = {}
                                            for block in blocks1:
                                                block_type = block.get('type', 'unknown')
                                                type_counts1[block_type] = type_counts1.get(block_type, 0) + 1

                                            st.write("**块类型分布:**")
                                            for block_type, count in sorted(type_counts1.items()):
                                                st.write(f"- {block_type}: {count}")

                                    with col2:
                                        st.markdown(f"#### {comp2['model_config']}")
                                        st.caption(f"解析时间: {comp2['timestamp']}")

                                        # 显示PDF预览
                                        if preview_img2:
                                            st.image(preview_img2, caption="Spans PDF预览", use_container_width=True)
                                        else:
                                            st.warning("⚠️ PDF预览生成失败")

                                        # 提供下载按钮
                                        st.download_button(
                                            "💾 下载完整Spans PDF",
                                            data=spans_pdf2,
                                            file_name=f"{selected_file.rsplit('.', 1)[0]}_{comp2['layout_model']}_{comp2['content_model']}_spans.pdf",
                                            mime="application/pdf",
                                            key="download_pdf2"
                                        )

                                        # 显示统计信息
                                        with st.expander("📊 查看统计信息"):
                                            blocks2 = comp2['result'].get('blocks', [])
                                            st.metric("总块数", len(blocks2))

                                            type_counts2 = {}
                                            for block in blocks2:
                                                block_type = block.get('type', 'unknown')
                                                type_counts2[block_type] = type_counts2.get(block_type, 0) + 1

                                            st.write("**块类型分布:**")
                                            for block_type, count in sorted(type_counts2.items()):
                                                st.write(f"- {block_type}: {count}")

                                else:
                                    st.error("❌ Spans PDF生成失败，请查看日志")

    else:
        st.info("暂无结果,请上传文件并开始解析")

    # 页脚
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
        <p>JCB-OCR混合模型解析系统 | 支持Qwen, DeepSeek, Hunyuan, MinerU</p>
        <p>可自由组合布局识别和内容识别模型</p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
