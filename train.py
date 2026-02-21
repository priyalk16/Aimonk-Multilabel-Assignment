import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt

from dataset import AimonkDataset
from model import get_model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


dataset = AimonkDataset(
    image_folder="images",
    label_file="labels.txt",
    transform=transform
)

dataloader = DataLoader(dataset, batch_size=16, shuffle=True)


model = get_model()
model = model.to(device)


criterion = nn.BCEWithLogitsLoss(reduction='none')

optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-5
)


num_epochs = 5
iteration_numbers = []
training_losses = []
iteration = 0


for epoch in range(num_epochs):
    model.train()

    for images, labels, mask in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        mask = mask.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)
        loss = loss * mask

        valid_elements = mask.sum()
        loss = loss.sum() / (valid_elements + 1e-6)

        loss.backward()
        optimizer.step()

        iteration += 1
        iteration_numbers.append(iteration)
        training_losses.append(loss.item())

    print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {loss.item():.4f}")


torch.save(model.state_dict(), "models/aimonk_model.pth")


plt.plot(iteration_numbers, training_losses)
plt.xlabel("iteration_number")
plt.ylabel("training_loss")
plt.title("Aimonk_multilabel_problem")
plt.savefig("models/loss_curve.png")
plt.close()

print("Training completed.")