# app.py (前端界面代码，用于运行演示，不需要放进 LaTeX 论文里)
import streamlit as st
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torch.nn as nn
import os
import sys

# 导入你的模型和攻击逻辑
from models.unet import UNet
from scripts.attacks.pgd_seg import pgd_attack_on_segmentation

st.set_page_config(page_title="Medical Seg Robustness", layout="wide")
st.title("🛡️ Medical Image Segmentation: Adversarial Robustness Evaluator")


# 1. 缓存模型加载，防止显存爆炸 (对应论文里提到的缓存优化)
@st.cache_resource
def load_cached_model(model_type, ckpt_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 填入你 baseline 没有报错时的 base_ch (比如 8 或 32)
    BASE_CH = 8  # <-- 如果你之前用 32 没报错，这里就填 32

    if os.path.exists(ckpt_path):
        # 1. 先加载权重字典
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        if 'model_state_dict' in state:
            state = state['model_state_dict']
        elif 'state_dict' in state:
            state = state['state_dict']

        new_state = {k[7:] if k.startswith('module.') else k: v for k, v in state.items()}

        # 2. ！！！自动侦测是否使用了 Spectral Normalization ！！！
        # 检查权重字典的 key 里有没有包含 'weight_orig' 的
        uses_sn = any('weight_orig' in k for k in new_state.keys())

        # 3. 根据侦测结果动态初始化模型
        model = UNet(in_channels=3, n_classes=2, base_ch=BASE_CH, spectral_norm=uses_sn).to(device)

        try:
            model.load_state_dict(new_state, strict=True)
            print(f"[{model_type}] Loaded successfully. Auto-detected SN: {uses_sn}")
        except RuntimeError as e:
            st.error(f"模型权重加载失败！\n{e}")
            raise e
    else:
        st.error(f"找不到权重文件：{ckpt_path}")
        # 如果找不到文件，返回一个默认模型防崩溃
        model = UNet(in_channels=3, n_classes=2, base_ch=BASE_CH, spectral_norm=False).to(device)

    model.eval()
    return model, device


# 2. 侧边栏：控制面板
st.sidebar.header("⚙️ Configuration")
model_choice = st.sidebar.radio(
    "Model Type",
    ["Baseline (Vulnerable)", "TRADES (Robust)"]
)
# 假设你的模型路径 (根据你的实际路径修改)
ckpt_dict = {
    "Baseline (Vulnerable)": "outputs/Covid/2000-fgsm/unet/unet_best_denoise.pth",
    "TRADES (Robust)": "outputs/Covid/2000-fgsm/adv/unet_adv_trained.pth"
}
model, device = load_cached_model(model_choice, ckpt_dict[model_choice])

st.sidebar.subheader("Attack Parameters")
epsilon = st.sidebar.slider("Epsilon (Perturbation Bound)", min_value=0.0, max_value=16.0, value=4.0, step=1.0)
iters = st.sidebar.slider("PGD Iterations", min_value=1, max_value=20, value=10, step=1)

# 3. 主界面：图片上传
uploaded_file = st.file_uploader("Upload a Medical Image (PNG/JPG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    transform = transforms.Compose([transforms.Resize((384, 512)), transforms.ToTensor()])
    img_t = transform(image).unsqueeze(0).to(device)

    # 我们生成一个假的空白 Mask 用来诱导攻击 (因为黑盒演示用户可能不传GT)
    # 真实场景中，攻击旨在让预测结果偏离“原本的预测”或“真实的标签”
    dummy_mask = torch.zeros((1, 384, 512), dtype=torch.long).to(device)

    st.write("### Real-Time Evaluation")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(image, use_container_width=True, caption="Original Image")

    with torch.no_grad():
        clean_logits = model(img_t)
        clean_pred = torch.argmax(torch.softmax(clean_logits, dim=1), dim=1).squeeze().cpu().numpy()

        # --- 消除边界伪影 (Clear Padding Artifacts) ---
        b_size = 15  # 切除最外围 15 个像素的伪影
        clean_pred[:b_size, :] = 0
        clean_pred[-b_size:, :] = 0
        clean_pred[:, :b_size] = 0
        clean_pred[:, -b_size:] = 0

    with col2:
        # 简单将预测Mask变成红色的图像叠加
        viz_clean = np.array(image.resize((512, 384))) / 255.0
        viz_clean[clean_pred > 0] = viz_clean[clean_pred > 0] * 0.5 + np.array([1, 0, 0]) * 0.5
        st.image(np.clip(viz_clean, 0, 1), use_container_width=True, caption="Clean Prediction")

    if st.button("Generate Adversarial Attack 🚀"):
        with st.spinner("Calculating Gradients and Attacking..."):
            criterion = nn.CrossEntropyLoss()
            # 执行攻击
            adv_img = pgd_attack_on_segmentation(
                model, img_t, dummy_mask,  # 实际上应该用 clean_pred 当作目标，或者破坏它
                eps=epsilon / 255.0, alpha=(epsilon / iters * 1.5) / 255.0, iters=iters,
                loss_fn=criterion, device=device
            )

            with torch.no_grad():
                adv_logits = model(adv_img)
                adv_pred = torch.argmax(torch.softmax(adv_logits, dim=1), dim=1).squeeze().cpu().numpy()

                # --- 消除边界伪影 (Clear Padding Artifacts) ---
                b_size = 15
                adv_pred[:b_size, :] = 0
                adv_pred[-b_size:, :] = 0
                adv_pred[:, :b_size] = 0
                adv_pred[:, -b_size:] = 0

            with col3:
                viz_adv = np.array(image.resize((512, 384))) / 255.0
                viz_adv[adv_pred > 0] = viz_adv[adv_pred > 0] * 0.5 + np.array([1, 0, 0]) * 0.5
                st.image(np.clip(viz_adv, 0, 1), use_container_width=True,
                         caption=f"Prediction Under Attack (eps={epsilon})")

        st.success("Attack Completed! Observe the degradation (or robustness) in the rightmost panel.")