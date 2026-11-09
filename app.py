import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


# 1. モデルの準備
@st.cache_resource
def load_model():
    model = models.mobilenet_v2()
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    model.load_state_dict(torch.load("mushroom_model.pt", map_location=torch.device("cpu")))
    model.eval()
    return model


model = load_model()

# ターゲット層の指定（MobileNetV2の最終層）
target_layers = [model.features[-1]]

# Grad-CAMのカメラをセットアップ
cam = GradCAM(model=model, target_layers=target_layers)

# 2. 画像の変換ルール
transform = transforms.Compose(
    [transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
)

# 日本語のクラス名リスト（アルファベット順：0=edible, 1=poisonous）
class_names = ["食用キノコ 🍄", "毒キノコ ☠️"]

# 3. 画面の作成と画像アップロード
st.title("キノコ判定AI 🍄")
uploaded_file = st.file_uploader("キノコの画像をアップロードしてください", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="アップロードされた画像", use_container_width=True)

    # 4. 予測と確率の計算
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_tensor)

        # 1. AIの注目度（ヒートマップ）を計算する
        # targets=None にすると、AIが最も確率が高いと判定したキノコに対する注目度を自動計算します
        grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]

        # 2. 元画像を合成用に準備する
        # 画像サイズをAIの入力サイズ（MobileNetV2なら通常 224x224）に合わせ、0.0〜1.0の数値に変換します
        rgb_img = np.array(image.convert("RGB"))
        rgb_img = cv2.resize(rgb_img, (224, 224))
        rgb_img = np.float32(rgb_img) / 255

        # 3. ヒートマップと元画像を合成する
        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        # 4. Streamlitで画面に表示する
        st.write("🧠 AIが注目したポイント")
        st.image(cam_image, use_container_width=True)

        # ① 結論（0番か1番か）を取得
        result_index = torch.argmax(outputs).item()

        # ② 確率を計算
        probabilities = F.softmax(outputs, dim=1)

        # 選ばれたクラスの確率をパーセント(%)に変換
        confidence = probabilities[0][result_index].item() * 100

    # 5. 結果の表示
    result_text = class_names[result_index]
    st.success(f"判定結果: **{result_text}**")
    st.info(f"AIの自信度: {confidence:.1f}%")
