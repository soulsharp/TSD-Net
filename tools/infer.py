import torch
import argparse
import os
from PIL import Image
from torchvision.transforms import v2 

from utils.utils import load_yaml, test, split_dataset
from model.classifier import TSD_Classifier
from dataset.dataset import DatasetFromFolder
from utils.utils import build_dataloader, build_criterion

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def inference(args):
    """
    Runs inference using a pre-trained TSD_Classifier model on the test split of the dataset.

    Workflow:
        - Loads the experiment configuration from a YAML file.
        - Retrieves the best threshold (determined during training) from the config.
        - Builds the TSD_Classifier model architecture and loss criterion.
        - Loads the best saved model checkpoint (best_model.pth).
        - Reconstructs the full dataset and reproducibly re-splits it to obtain the test set.
        - Builds a test DataLoader and evaluates the model using the stored best threshold.
        - Computes and prints evaluation metrics (loss, accuracy, precision, recall, F1-score).

    Args:
        args (argparse.Namespace): Parsed command-line arguments. Expected fields:
            - config_path (str): Path to the YAML configuration file.
            - num_classes (int): Number of classes for classification.

    Raises:
        ValueError: If the config file is missing or invalid.
        FileNotFoundError: If the best model checkpoint is missing.

    Effects:
        - Loads and evaluates the trained model on the test set.
        - Outputs evaluation metrics to stdout using the optimal threshold.
    """

    # Reads the config file #
    config = load_yaml(args.config_path)
    print("Training config: \n", config)
    
    if config is None:
        raise ValueError(f"Failed to load config file: {args.config_path}")
    
    # Config params
    dataset_config = config['dataset_params']
    tsd_config = config['tsd_params']
    train_config = config['train_params']
    threshold = train_config["best_threshold"]

    model = TSD_Classifier(args.num_classes, tsd_config["emb_dim"], 
                           tsd_config["num_heads_tiny"], tsd_config["num_encoder_layers_tiny"],
                           tsd_config["expansion_ratio"], tsd_config["cte_output_channels"])
    
    # model = TSD_Classifier(args.num_classes, tsd_config["emb_dim"], 
    #                        tsd_config["num_heads_big"], tsd_config["num_encoder_layers_big"],
    #                        tsd_config["expansion_ratio"], tsd_config["cte_output_channels"])
    
    criterion = build_criterion().to(device)
    model = model.to(device)
    criterion = criterion.to(device)

    model_file_path = os.path.join(train_config["checkpoint_dir"], "best", "best_model.pth")

    if not os.path.exists(model_file_path):
        raise FileNotFoundError(f"Model checkpoint not found at {model_file_path}")
    try:
        state_dict = torch.load(model_file_path, map_location="cpu")
    except Exception as e:
        raise RuntimeError(f"Failed to load model checkpoint: {e}")

    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    
    # This block tests custom images passed in at inference time
    if args.image_path:
        # Uses the default image transformation, not a permanent solution if actual transformation of the inputs was different
        transform = v2.Compose([
            v2.Resize((224, 224), interpolation=v2.InterpolationMode.BILINEAR),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True)
            ])

        results = run_on_images(model, args.image_path, threshold, transform, device)
        for fname, pred in results:
            print(f"{fname} -> Predicted label: {pred}")
    
    else:

        # This block tests held-out data in the original dataset
        im_dataset = DatasetFromFolder(dataset_config["im_path"], dataset_config["classification_only"])
        _, _, test_dataset = split_dataset(im_dataset, train_config,
                                                    dataset_config["split_file_path"], dataset_config["split_seed"]
                                                )
        
        test_loader = build_dataloader(test_dataset, train_config, is_train=False)

        test(test_loader, model, criterion, threshold)

@torch.no_grad()
def run_on_images(model, image_path, threshold, transform, device):
    """
    Runs the model on a single image or a folder of images.

    Args:
        model (nn.Module): The trained model.
        image_path (str): Path to a single image or directory of images.
        threshold (float): Threshold for binary classification.
        transform (callable): Image transformation pipeline.
        device (torch.device): Target device for inference.

    Returns:
        List of tuples: (filename, predicted_label)
    """
    model.eval()
    model.to(device)

    if os.path.isdir(image_path):
        img_files = [os.path.join(image_path, f) for f in os.listdir(image_path)
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    else:
        img_files = [image_path]

    results = []
    for img_file in img_files:
        image = Image.open(img_file).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device)

        logits = model(input_tensor, return_logits=True)
        pred = (logits > threshold).int()

        if pred.ndim == 2 and pred.shape[1] == 1:
            pred = pred.squeeze(1)
        results.append((os.path.basename(img_file), pred.item()))

    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference settings')
    parser.add_argument('--config', dest='config_path',
                        default='config/config.yaml', type=str)
    parser.add_argument("--num_classes", default=2, type=int)
    parser.add_argument('--image_path', type=str, default=None,
                    help='Path to image or folder of images for custom inference')
    args = parser.parse_args()
    inference(args)