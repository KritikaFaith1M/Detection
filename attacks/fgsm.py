import torch
import torch.nn.functional as F

class FGSMAttack:
    def __init__(self, model, device="cpu", epsilon=0.2):
        self.model = model
        self.device = device
        self.epsilon = epsilon

    def generate(self, image, label):
        image = image.to(self.device)
        label = label.to(self.device)

        image.requires_grad = True

        # -------------------------------
        # FORWARD PASS
        # -------------------------------
        output = self.model(image)
        loss = F.cross_entropy(output, label)

        # -------------------------------
        # BACKWARD PASS
        # -------------------------------
        self.model.zero_grad()
        loss.backward()

        # -------------------------------
        # FGSM PERTURBATION
        # -------------------------------
        gradient = image.grad.data
        perturbed_image = image + self.epsilon * gradient.sign()

        # CLAMP (IMPORTANT)
        perturbed_image = torch.clamp(perturbed_image, -1, 1)

        return perturbed_image.detach()