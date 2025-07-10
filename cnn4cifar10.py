import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt

class LeNet5Cifar(nn.Module):
    def __init__(self, num_class):
        super(LeNet5Cifar, self).__init__()
        self.cnn_level = nn.Sequential(
            nn.Conv2d(3, 6, 5), # input channels, output channels, kernel size
            nn.Sigmoid(),
            nn.AvgPool2d(2, 2),  # kernel size, stride, padding = 0 (default)
            nn.Conv2d(6, 16, 5),
            nn.Sigmoid(),
            nn.AvgPool2d(2,2),
            nn.Conv2d(16,120,5),
            nn.Sigmoid(),
        )

        self.fc_level = nn.Sequential(
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, num_class),
        )

    def forward(self, x):
        x = self.cnn_level(x)
        x = torch.flatten(x, 1)
        x = self.fc_level(x)
        prob = F.log_softmax(x, dim=1)
        return prob

def load_data(batch_size=256):
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
    return trainloader, testloader

def train_and_evaluate(model, trainloader, testloader, device, num_epochs=20, lr=0.001):
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr,momentum=0.9)
    
    train_loss = []
    test_loss = []
    train_acc = []
    test_acc = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_loss.append(running_loss / len(trainloader))
        train_acc.append(correct_train / total_train)

        # Evaluate on test set
        model.eval()
        test_running_loss = 0.0
        correct_test = 0
        total_test = 0

        with torch.no_grad():
            for inputs, labels in testloader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()

        test_loss.append(test_running_loss / len(testloader))
        test_acc.append(correct_test / total_test)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss[-1]:.4f}, Test Loss: {test_loss[-1]:.4f}, "
              f"Train Acc: {train_acc[-1]:.4f}, Test Acc: {test_acc[-1]:.4f}")
        
    print('Finished Training')
    
    return train_loss, test_loss, train_acc, test_acc

if __name__ == "__main__":
    print(torch.__version__)
    print(torch.cuda.is_available())
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    model = LeNet5Cifar(len(classes)).to(device)
    trainloader, testloader = load_data()

    train_loss, test_loss, train_acc, test_acc = train_and_evaluate(model, trainloader, testloader, device)

    torch.save(model.state_dict(), "lenet5_cifar.pth")
    print("Model saved as 'lenet5_cifar.pth'")

    # Plotting the results
    plt.figure(figsize=(12, 6))

    # Loss vs Epoch
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='Train Loss')
    plt.plot(test_loss, label='Test Loss')
    plt.title('Loss vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Accuracy vs Epoch
    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label='Train Accuracy')
    plt.plot(test_acc, label='Test Accuracy')
    plt.title('Accuracy vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Loss & Accuracy
    plt.figure(figsize=(12, 6))

    epochs = range(1, len(train_loss) + 1)

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Loss (왼쪽 Y축)
    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(epochs, train_loss, label='Train Loss', color='red', linestyle='--')
    ax1.plot(epochs, test_loss, label='Test Loss', color='orange', linestyle='-')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.legend(loc='upper left')

    # Accuracy (오른쪽 Y축)
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Accuracy', color=color)
    ax2.plot(epochs, train_acc, label='Train Accuracy', color='blue', linestyle='--')
    ax2.plot(epochs, test_acc, label='Test Accuracy', color='green', linestyle='-')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend(loc='upper right')

    # Title and layout
    plt.title('Loss & Accuracy vs Epoch')
    fig.tight_layout()
    plt.show()
    """