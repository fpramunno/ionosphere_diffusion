from sklearn.metrics import roc_auc_score
import torch
from torch.nn import functional as F
import torch.nn as nn
import torchvision.models as models


class PerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG16 features.
    Compares images in feature space to preserve high-frequency details.
    """
    def __init__(self, layer_idx=16, device='cuda'):
        super().__init__()
        # Use first 16 layers of VGG16 (up to relu3_3)
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features[:layer_idx]
        vgg = vgg.to(device).eval()
        for p in vgg.parameters():
            p.requires_grad = False
        self.vgg = vgg
        # ImageNet normalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, recon, target):
        # Handle grayscale: repeat to 3 channels
        if recon.shape[1] == 1:
            recon = recon.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)

        # Normalize to ImageNet stats
        recon = (recon - self.mean.to(recon.device)) / self.std.to(recon.device)
        target = (target - self.mean.to(target.device)) / self.std.to(target.device)

        # Extract features and compute loss
        recon_features = self.vgg(recon)
        target_features = self.vgg(target)

        return F.mse_loss(recon_features, target_features)

def mean_accuracy(targets, logits):
        """
        Evaluates the mean accuracy for multi-class classification.

        Args:
            targets: ground truth class indices (0, 1, or 2)
            logits: raw model outputs (before softmax), shape [batch, num_classes]
        """
        # Get predicted class (argmax of logits)
        predictions = torch.argmax(logits, dim=1)
        targets = targets.view(-1).long()

        # Compute accuracy
        correct = (predictions == targets).float()
        acc = correct.sum() / targets.size(0)

        # Compute multi-class AUC (one-vs-rest)
        score_roc_auc = 0.0
        try:
            probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
            targets_np = targets.detach().cpu().numpy()
            score_roc_auc = roc_auc_score(targets_np, probs, multi_class='ovr', average='macro')
        except ValueError:
            pass

        return acc, score_roc_auc
    
def reconstruction_loss(recon_x, x, recon_param , dist):
    BCE = torch.nn.BCELoss(reduction="sum") 
    batch_size = recon_x.shape[0]
    if dist == 'bernoulli':

            recons_loss = BCE(recon_x, x) / batch_size
    elif dist == 'gaussian':
            x_recons = recon_x
            recons_loss = F.mse_loss(x_recons, x, reduction='sum') /batch_size
    else:
        raise AttributeError("invalid dist")
    return recon_param * recons_loss

def KL_loss(mu, logvar, z_dist, prior_dist, beta, c=0.0):

    # KL divergence loss
    KLD_1 = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    ############################

    KLD_2 = torch.distributions.kl.kl_divergence(z_dist, prior_dist)
    KLD_2 = KLD_2.sum(1).mean()
    KLD_2 = beta * (KLD_2 - c).abs()
    return beta * KLD_1, KLD_2


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    Compute KL divergence between two Gaussians.
    Source: https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/losses.py

    KL(N(mean1, var1) || N(mean2, var2))

    Works with any tensor shape - broadcasts automatically.
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, torch.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # Force variances to be Tensors
    logvar1, logvar2 = [
        x if isinstance(x, torch.Tensor) else torch.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + torch.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
    )


def autoencoder_kl_loss(mu, logvar, beta, c=0.0):
    """
    KL loss for AutoencoderKLAttri (and other VAEs with flat mu/logvar).

    Uses closed-form KL divergence against standard normal prior N(0,1).

    Args:
        mu: mean tensor, shape (B, latent_dim) - can be flat or will be flattened
        logvar: log variance tensor, same shape as mu
        beta: weight for KL term (beta-VAE)
        c: capacity term for controlled capacity increase (default 0)

    Returns:
        kl_loss: scalar KL divergence loss
    """
    # Flatten if spatial: (B, C, H, W) -> (B, C*H*W)
    if mu.dim() > 2:
        mu = mu.view(mu.size(0), -1)
        logvar = logvar.view(logvar.size(0), -1)

    # KL against standard normal prior (mean=0, logvar=0)
    kl_elementwise = normal_kl(mu, logvar, 0, 0)

    # Sum over latent dims, mean over batch
    kl_loss = kl_elementwise.sum(dim=1).mean()

    # Apply beta weight and optional capacity constraint
    kl_loss = beta * (kl_loss - c).abs()

    return kl_loss


def mlp_loss_function(y, out_mlp, alpha):
    """
    Multi-class classification loss using CrossEntropyLoss.

    Args:
        y: target class indices (0, 1, or 2), shape [batch, 1] or [batch]
        out_mlp: raw logits from model, shape [batch, num_classes]
        alpha: loss weight multiplier
    """
    criterion = torch.nn.CrossEntropyLoss()
    targets = y.view(-1).long()  # CrossEntropyLoss expects [batch] of class indices
    mean_loss = criterion(out_mlp, targets)

    return alpha * mean_loss

def reg_loss_sign(latent_code, attribute, factor=1.0):
        # compute latent distance matrix
        latent_code = latent_code.view(-1, 1).repeat(1, latent_code.shape[0])
        #print(f"latent code shape: {latent_code.shape}")
        lc_dist_mat = (latent_code - latent_code.transpose(1, 0)).view(-1, 1)

        # compute attribute distance matrix
        attribute = attribute.view(-1, 1).repeat(1, attribute.shape[0])
        #print(attribute.shape)
        attribute_dist_mat = (attribute - attribute.transpose(1, 0)).view(-1, 1)

        # compute regularization loss
        loss_fn = torch.nn.L1Loss()
        lc_tanh = torch.tanh(lc_dist_mat * factor) # factor: tunable hyperparameter
        attribute_sign = torch.sign(attribute_dist_mat)
        sign_loss = loss_fn(lc_tanh, attribute_sign.float())

        return sign_loss

def reg_loss(latent_code, radiomics_, mini_batch_size, gamma = 1.0, factor = 1.0):
    AR_loss = 0.0
    for dim in range(radiomics_.shape[1]):
        x = latent_code[:, dim]
        radiomics_dim = radiomics_[:, dim]    
        AR_loss += reg_loss_sign(x, radiomics_dim, factor=factor )
    return gamma * AR_loss