import torch
import torch.nn as nn
import timm

def build_vit_tiny(
    num_classes=3,
    pretrained=True,
    image_size=224,
    dropout: float = 0.0
):
    """
    Builds a ViT-Tiny model using timm.
    Uses a Sequential head: Dropout + Linear
    to match the training checkpoint structure:
        head.0 = Dropout
        head.1 = Linear
    """
    model_name = f"vit_tiny_patch16_{image_size}"

    model = timm.create_model(model_name, pretrained=pretrained)

    in_features = model.head.in_features

    # ALWAYS use Sequential head (even if dropout=0)
    model.head = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes)
    )

    return model
