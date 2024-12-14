# extract_features.py
import os
import torch
from torchvision import transforms
from PIL import Image
import pickle
import torch.nn as nn
import torchvision


# Định nghĩa lại model class
class MetricFeatureExtractor(nn.Module):
    def __init__(self, embedding_dim=2048):
        super().__init__()
        resnet = torchvision.models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.projection = nn.Sequential(
            nn.Linear(2048, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU()
        )

    def forward(self, x):
        features = self.backbone(x)
        features = self.global_pool(features)
        features = features.view(features.size(0), -1)
        embeddings = self.projection(features)
        return embeddings

    def extract_features(self, images):
        return self.forward(images)


# Hàm trích xuất đặc trưng từ ảnh
def extract_features_from_images(model, dataset_dir, transform):
    features = []
    image_paths = []

    # Duyệt qua tất cả thư mục con
    for class_dir in os.listdir(dataset_dir):
        class_path = os.path.join(dataset_dir, class_dir)
        if os.path.isdir(class_path):
            print(f"Processing {class_dir}...")
            for img_name in os.listdir(class_path):
                if img_name.endswith(('.jpg', '.png', '.jpeg')):
                    img_path = os.path.join(class_path, img_name)
                    try:
                        img = Image.open(img_path).convert('RGB')
                        img_tensor = transform(img).unsqueeze(0)
                        with torch.no_grad():
                            feature = model.extract_features(img_tensor).numpy()
                            features.append(feature[0])
                            image_paths.append(img_path)
                    except Exception as e:
                        print(f"Error processing {img_path}: {str(e)}")

    return features, image_paths


def main():
    # Định nghĩa transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Đường dẫn
    MODEL_PATH = 'metric_model.pth'
    DATASET_DIR = 'static/dataset/Mon_An_Ha_Noi'

    # Load model
    print("Loading model...")
    model = MetricFeatureExtractor()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()

    # Trích xuất đặc trưng
    print("Extracting features...")
    features, image_paths = extract_features_from_images(model, DATASET_DIR, transform)

    # Lưu features và paths
    print("Saving features and paths...")
    with open("features_and_paths.pkl", "wb") as f:
        pickle.dump({
            "features": features,
            "image_paths": image_paths
        }, f)

    print(f"Done! Processed {len(image_paths)} images")
    print(f"Features shape: {len(features)}x{len(features[0])}")


if __name__ == "__main__":
    main()