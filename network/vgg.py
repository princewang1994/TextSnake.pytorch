import torchvision.models.vgg as vgg
import torch.nn as nn

class VGG16(nn.Module):

    def __init__(self):
        super().__init__()
        self.net = vgg.vgg16(pretrained=True)

        self.stage1 = nn.Sequential(*[self.net.features[layer] for layer in range(0, 5)])
        self.stage2 = nn.Sequential(*[self.net.features[layer] for layer in range(5, 10)])
        self.stage3 = nn.Sequential(*[self.net.features[layer] for layer in range(10, 17)])
        self.stage4 = nn.Sequential(*[self.net.features[layer] for layer in range(17, 24)])
        self.stage5 = nn.Sequential(*[self.net.features[layer] for layer in range(24, 31)])

    def forward(self, x):
        C1 = self.stage1(x)
        C2 = self.stage2(C1)
        C3 = self.stage3(C2)
        C4 = self.stage4(C3)
        C5 = self.stage5(C4)
        return C1, C2, C3, C4, C5


if __name__ == '__main__':
    import torch
    input = torch.randn((4, 3, 512, 512))
    net = VGG16()
    net(input)
