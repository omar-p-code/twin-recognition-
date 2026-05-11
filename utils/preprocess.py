import torchvision.transforms as transforms
from config import IMG_SIZE, NORMALIZE_MEAN, NORMALIZE_STD

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
])


def preprocess_image(img):
    """Convert a PIL image to a model-ready (1, C, H, W) tensor."""
    if img is None:
        return None
    try:
        img = img.convert("RGB")
        return transform(img).unsqueeze(0)
    except Exception as e:
        print(f"Preprocess error: {e}")
        return None
