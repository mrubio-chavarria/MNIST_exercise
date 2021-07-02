#!/home/mario/Projects/MNIST_exercise/venv/bin python

# Libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torch.multiprocessing as mp


# Libraries
def get_n_params(model):
    """
    DESCRIPTION:
    Function to count number of parameters. 
    """
    np=0
    for p in list(model.parameters()):
        np += p.nelement()
    return np


# Classes
class FCN(nn.Module):
    """
    DESCRIPTION:
    The class that decribes the fully-connected neural network.
    """
    # Methods
    def __init__(self, input_size, n_hidden, output_size):
        """
        DESCRIPTION:
        Constructor. 
        :param input_size: [int] number of pixels in your image.
        :param n_hidden: [int] number of neurons in the hidden layer.
        :param output_size: [int] number of classes to predict.
        """
        super(FCN, self).__init__()
        self.input_size = input_size
        self.n_hidden = n_hidden
        self.output_size = output_size
        self.network = nn.Sequential(
            nn.Linear(input_size, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, output_size),
            # Commented because of the nn.CrossEntropyLoss
            # nn.LogSoftmax(dim=1)
        )

    def forward(self, input_image):
        """
        DESCRIPTION:
        Method to compute the network output.
        :param input_image: [numpy.ndarray] the  image to classify.
        """
        input_image = input_image.view(-1, self.input_size)
        return self.network(input_image)


class CNN(nn.Module):
    """
    DESCRIPTION:
    The class that describes the convolutional neural network.
    """
    # Methods
    def __init__(self, input_size, n_features, output_size, kernel_size):
        """
        DESCRIPTION: 
        Constructor.
        :param input_size: [int] one dimension of the square images.
        :param n_hidden: [int] number of features to look for.
        :param output_size: [int] number of classes to predict.
        :param kernel_size: [int] size of the kernel to perform the
        convolution.
        """
        super(CNN, self).__init__()
        self.input_size = input_size
        self.network = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=n_features, kernel_size=kernel_size),
            nn.BatchNorm2d(n_features),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=n_features, out_channels=n_features, kernel_size=kernel_size),
            nn.BatchNorm2d(n_features),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(16 * n_features, 16),
            nn.ReLU(),
            nn.Linear(16, output_size),
            # Commented because of the nn.CrossEntropyLoss
            # nn.LogSoftmax(dim=1)
        )
    
    def forward(self, input_image):
        """
        DESCRIPTION:
        Method to compute the network output.
        :param input_image: [numpy.ndarray] the  image to classify.
        """
        return self.network(input_image)


# Functions
def train(n_epochs, model, rank):
    """
    DESCRIPTION:
    Method to train the model.
    :param epoch: [int] indication of the current training epoch.
    :param model: [nn.Module] the model to train.
    """
    # Set trainin mode
    model.train()
    # Define optimiser
    optimiser = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    # Define loss
    loss_function = nn.CrossEntropyLoss()
    # Train
    accuracy_list = []
    for epoch in range(n_epochs):
        for batch_id, (data, target) in enumerate(train_loader):
            # Send to device
            data, target = data.to(device), target.to(device)
            # Clean the gradient
            optimiser.zero_grad()
            # 1. Forward pass
            output = model(data)
            # 2. Loss
            # loss = F.nll_loss(output, target)
            loss = loss_function(output, target)
            # 3. Backpropagation
            loss.backward()
            # 4. Go step in the gradient
            optimiser.step()
            # Print progress
            if batch_id % 100 == 0:
                print('Process: {} Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                rank+1, epoch, batch_id * len(data), len(train_loader.dataset),
                100. * batch_id / len(train_loader), loss.item()))
        # Test epoch outcome
        test(network, accuracy_list)


def test(model, accuracy_list):
    """
    DESCRIPTION:
    A function to train the model.
    :param model: [nn.Module] the model whose perfomance shoudl be assessed.
    """
    # Set testing mode
    model.eval()
    # Initialise parameters
    test_loss = 0
    correct = 0
    # Start testing
    for data, target in test_loader:
        # Send to device
        data, target = data.to(device), target.to(device)
        # Forward step
        output = model(data)
        # Compute the loss
        test_loss += F.nll_loss(output, target, reduction='sum').item()
        # Obtain index of the neuron with the highest score
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
    # Compute testing parameters
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    accuracy_list.append(accuracy)
    # Print test results
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        accuracy))


if __name__ == '__main__':

    # # Create an experiment with your api key
    # # To connect a project with comet, just copy this code and substitute
    # # the aPI variables to select the project
    # experiment = Experiment(
    #     api_key="rqM9qXHiO7Ai4U2cqj1pS4R2R",
    #     project_name="mnist-test",
    #     workspace="mrubio-chavarria",
    # )

    n_processes = 4

    # Select device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load MNIST database
    input_size  = 28*28   # Images are 28x28 pixels
    output_size = 10      # There are 10 classes

    # Train
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True, 
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                    ])),
        batch_size=64, shuffle=True)

    # Test
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
        batch_size=1000, shuffle=True)

    # # Create the FCN model
    # n_hidden = 256
    # network = FCN(input_size, n_hidden, output_size)
    # network.to(device)

    # Create the CNN model
    n_features = 12
    kernel_size = 5
    network = CNN(int(input_size ** 0.5), n_features, output_size, kernel_size)
    network.to(device)

    network.share_memory()

    # Start training
    n_epochs = 2
    print('Number of parameters: {}'.format(get_n_params(network)))
    print('START TRAINING')
    processes = []
    for rank in range(n_processes):
        p = mp.Process(target=train, args=(n_epochs, network, rank))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()        

    # Save the model
    path = './saved_models/model.json'
    torch.save(network.state_dict(), path)

    
    