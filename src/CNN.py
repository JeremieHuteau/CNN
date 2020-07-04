import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsummary

#from groupy.gconv import pytorch_gconv

import pytorch_model
import model_building as build

def conv_block(
        in_channels, out_channels, kernel_size, 
        stride=1, padding=0, dilation=1, groups=1, depthwise=False,
        dropout=0,
    ):
        layers = nn.ModuleList()
        layers.extend([
            nn.Conv2d(
                in_channels, out_channels if not depthwise else in_channels, 
                kernel_size, stride=stride, padding=padding, dilation=dilation,
                groups=groups if not depthwise else in_channels
            ),
        ])
        if depthwise:
            layers.extend([
                nn.Conv2d(in_channels, out_channels, 1, groups = groups),
            ])
        layers.extend([
            nn.ReLU(inplace=True),
        ])
        layers.extend([
            nn.BatchNorm2d(out_channels),
        ])
        if dropout > 0:
            layers.extend([
                nn.Dropout2d(dropout, inplace=False),
            ])


        #if groups > 1:
        #    layers.extend([
        #        build.ChannelShuffle(groups)
        #    ])
        #layers.extend([
        #    Excitation(out_channels)
        #])

        return nn.Sequential(*layers)

class Excitation(nn.Module):
    def __init__(self,
        in_channels,
    ):
        super(Excitation, self).__init__()
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.dense = nn.Linear(in_channels, in_channels)
        self.activation = nn.Sigmoid()
        self.multiply = build.Multiply()

    def forward(self, x):
        batch_size, channels, _, _ = x.size()

        excitation = self.pooling(x).view(batch_size, channels)
        excitation = self.dense(excitation)
        excitation = self.activation(excitation).view(
            batch_size, channels, 1, 1)

        x = self.multiply(x, excitation)

        return x


class ResidualBlock(nn.Module):
    def __init__(self,
            in_channels, out_channels, kernel_size,
            stride = 1, padding=None, **kwargs):
        super(ResidualBlock, self).__init__()

        if padding is None:
            padding = int(stride<2)*(kernel_size//2)

        self.stride = stride

        # Branch 1
        #if stride > 1:
        #    #conv_block(
        #    #    in_channels, out_split_channels, kernel_size,
        #    #    stride=stride, padding=0,#kernel_size//2, 
        #    #    depthwise=True
        #    #)
        #else:
        #    self.branch1 = nn.Sequential(
        #        nn.Conv2d(in_channels, out_channels, 1),
        #        nn.ReLU(),
        #        nn.BatchNorm2d(out_channels),
        #    )

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, 
                stride=stride, padding=padding),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, 
                stride=stride, padding=padding),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, 1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
        )

        self.add = build.Add()

    def forward(self, x):
        x = self.add(self.branch1(x), self.branch2(x))

        return x

class Transformer(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0, height=1):
        super(Transformer, self).__init__()
        self.mha = nn.ModuleList([
            nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
            for i in range(height)
        ])
        self.bn = nn.ModuleList([
            nn.BatchNorm1d(embed_dim)
            for i in range(height)
        ])
        self.activation = nn.ReLU()

    def forward(self, x):
        batch_size, channels, h, w = x.size()
        x = x.view(batch_size, channels, h*w)
        for mha, bn in zip(self.mha, self.bn):
            attended = torch.transpose(x, 0, 2)
            attended = torch.transpose(attended, 1, 2)
            attended, attention = mha(*[attended]*3)
            attended = self.activation(attended)

            attended = torch.transpose(attended, 2, 1)
            attended = torch.transpose(attended, 2, 0)
            attended = bn(attended)

            x = torch.add(x, attended)

        x = x.view(batch_size, channels, h, w)
        return x

class GroupView(nn.Module):
    def __init__(self):
        super(GroupView, self).__init__()

    def forward(self, x):
        shape = x.size()
        x = x.view(-1, shape[1]*shape[2], shape[3], shape[4])
        return x

#class GConvBlock(nn.Module):
#    def __init__(self,
#        in_channels, out_channels, kernel_size, 
#        stride=1, padding=0, dilation=1, groups=1, depthwise=False,
#        #dropout_p=0,
#    ):
#        super(GConvBlock, self).__init__()
#
#        self.in_channels = in_channels//4
#        self.out_channels = out_channels//4
#        self.groups = groups
#
#        self.group_convs = nn.ModuleList()
#        in_group_channels = self.in_channels // groups
#        out_group_channels = self.out_channels // groups
#        self.group_convs.extend([
#            pytorch_gconv.P4ConvP4(
#                in_group_channels, out_group_channels, kernel_size,
#                stride = stride, padding = padding)
#            for i in range(groups)
#        ])
#
#        self.block = nn.Sequential(
#            nn.ReLU(),
#            nn.BatchNorm3d(self.out_channels),
#            #GroupView(),
#        )
#
#    def forward(self, x):
#        if self.groups > 1:
#            chunks = x.chunk(self.groups, dim=1)
#            x = torch.stack([
#                group_conv(chunk) 
#                for (group_conv, chunk) in zip(self.group_convs, chunks)
#            ], dim=1)
#            x = torch.transpose(x, 1, 2).contiguous()
#            h, w = x.size()[-2:]
#            x = x.view(-1, self.out_channels, 4, h, w)
#        else:
#            x = self.group_convs[0](x)
#        return self.block(x)

class ReusableConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
            padding=0, nb_use=1):
        super(ReusableConvBlock, self).__init__()
        self.nb_use = nb_use

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.ReLU(),
        )

        self.bn = nn.ModuleList([
            nn.BatchNorm2d(out_channels)
            for i in range(self.nb_use)
        ])

    def forward(self, x, current_use=0):
        x = self.block(x)
        x = self.bn[current_use](x)

        return x

class DenseConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
            depth, k=12, padding=0, groups=1):
        super(DenseConvBlock, self).__init__()
        print("depth=", depth)

        inner_kernel_size = kernel_size - ((kernel_size+1)%2)
        inner_padding = inner_kernel_size // 2

        self.conv_layers = nn.ModuleList([
            self._conv_block(in_channels + i*k, k, kernel_size,
                padding=inner_padding, groups=groups)
            for i in range(depth)
        ] + [
            self._conv_block(in_channels+(depth*k), out_channels, 1,
                groups=groups)
        ])
        
    def forward(self, x):
        for layer in self.conv_layers[:-1]:
            x = torch.cat([x, layer(x)], dim=1)
        feature_maps = x
        x = self.conv_layers[-1](x)
        return x, feature_maps

    def _conv_block(self, in_channels, out_channels, kernel_size, 
            padding=0, groups=1):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, 
                padding=padding, groups=groups),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
        )

class DenseNet(nn.Module):
    def __init__(self, in_channels, out_channels, adaptive_pooling_size=1,
            **kwargs):
        super(DenseNet, self).__init__()
        self.stages = nn.ModuleList()

        nb_stages = 5
        nb_dense_blocks = 3
        stage_strides = [1, 2, 2, 1, 1]

        width_delta = (out_channels - in_channels) 
        print("width_delta", width_delta)
        first_stages_out_channels = in_channels + int(width_delta * 0.3)
        print("first_stages_out_channels", first_stages_out_channels)
        first_stages_out_channels += 4-(first_stages_out_channels%4)
        print("first_stages_out_channels", first_stages_out_channels)
        stage_widths = [in_channels] + [
            build.channel_sizing_linear(in_channels, first_stages_out_channels, 
                i+1, 3)[0]
            for i in range(0, 3)
        ] + [
            build.channel_sizing_linear(first_stages_out_channels, out_channels, 
                i+1, 2)[0]
            for i in range(0, 1)
        ] + [out_channels]

        print(stage_widths)

        # First two are normal blocks, last are DenseNet blocks
        stage_lengths = [1]*(nb_stages-nb_dense_blocks) + [2]*nb_dense_blocks
        stage_depths = [0, 0, 0, 0, 3]
        stage_k = [0, 0, 0, 16, 32]
        stage_groups = [1, 1, 1, 1, 2]

        self.nb_agregate_channels = 0

        basic_block = conv_block
        for i in range(nb_stages-nb_dense_blocks):
            self.stages.append(nn.ModuleList([
                basic_block(*stage_widths[i:i+2], 3, 
                    stride=stage_strides[i], groups=stage_groups[i])
            ]))
            nb_agregate_channels = stage_widths[i]
            self.nb_agregate_channels += nb_agregate_channels

            #self.stages.append(build.Stage(
            #    basic_block, 
            #    *stage_widths[i:i+2], 3, 
            #    stride=stage_strides[i],
            #    length=stage_lengths[i],
            #    channel_sizing='linear'
            #))

        for i in range(nb_stages-nb_dense_blocks, nb_stages-2):

            depth = stage_depths[i]#-(nb_stages-nb_dense_blocks)]
            k = stage_k[i]#-(nb_stages-nb_dense_blocks)]

            self.stages.append(nn.ModuleList([
                #DenseConvBlock(
                #    *stage_widths[i:i+2], 3, 
                #    depth, k=k, groups=stage_groups[i]),
                basic_block(
                    stage_widths[i], stage_widths[i+1], 3,
                    stride=stage_strides[i])

            ]))
            nb_agregate_channels = stage_widths[i] #\ 
                #+ stage_widths[i+1] + depth*k
            self.nb_agregate_channels += nb_agregate_channels
            #self.stages.append(build.Stage(
            #    build.block_factory(DenseConvBlock, 
            #        stage_depths[i-(nb_stages-nb_dense_blocks)]),
            #    *stage_widths[i:i+2], 3, 
            #    pooling_block=basic_block,
            #    stride=stage_strides[i],
            #    length=stage_lengths[i],
            #    channel_sizing='linear'
            #))

        self.avg_pool = nn.AdaptiveAvgPool2d(adaptive_pooling_size)
        self.max_pool = nn.AdaptiveMaxPool2d(adaptive_pooling_size)

        self.skip_avg5_pool = nn.AdaptiveAvgPool2d(5)
        self.skip_avg3_pool = nn.AdaptiveAvgPool2d(3)
        self.skip_max5_pool = nn.AdaptiveMaxPool2d(5)
        self.skip_max3_pool = nn.AdaptiveMaxPool2d(3)

        for i in range(nb_stages-2,nb_stages):
            depth = stage_depths[i]
            k = stage_k[i]

            middle_block = basic_block(
                    stage_widths[i+1]*2, stage_widths[i+1], 3, 
                    padding=1
            ) if stage_depths[i] < 2 else DenseConvBlock(
                    stage_widths[i+1]*2, stage_widths[i+1], 3, 
                    depth, k=k, groups=stage_groups[i])

            self.stages.append(nn.ModuleList([
                basic_block(
                    self.nb_agregate_channels+stage_widths[i], stage_widths[i+1]*2,
                    1),
                middle_block,
                #basic_block(
                #    stage_widths[i+1]*2, stage_widths[i+1], 3, padding=1),
                #DenseConvBlock(
                #    stage_widths[i+1]*2, stage_widths[i+1], 3, 
                #    depth, k=k, groups=stage_groups[i]),
                basic_block(
                    stage_widths[i+1]*1, stage_widths[i+1], 3,
                    stride=stage_strides[i])

            ]))
            nb_agregate_channels = stage_widths[i] + stage_widths[i+1]*3 + \
                    depth*k*int(stage_depths[i] > 1)
            self.nb_agregate_channels += nb_agregate_channels

        #nb_agregate_channels = sum([
        #    build.channel_sizing_linear(
        #        in_channels, out_channels,
        #        i, nb_stages)[0]
        #    for i in range(nb_stages)
        #]) + in_channels
        #self.nb_agregate_channels = sum(stage_widths[:-1])
        self.nb_agregate_channels *= 1 * (adaptive_pooling_size**2)
        #self.nb_agregate_channels += 144


    def forward(self, x):
        feature_maps = []

        for stage in self.stages[:-2]:
            for block in stage:
                block_output = block(x)

                if isinstance(block_output, tuple):
                    x, feature_map = block_output
                else:
                    x, feature_map = block_output, x

                feature_maps.extend([
                    self.skip_max5_pool(feature_map),
                ])
                #features.extend([
                #    self.avg_pool(feature_map), 
                #    self.max_pool(feature_map)
                #])


        features = []
        for i, stage in enumerate(self.stages[-2:]):
            x = torch.cat(feature_maps+[x], dim=1)
            feature_maps = []

            for block in stage:
                block_output = block(x)

                if isinstance(block_output, tuple):
                    x, feature_map = block_output
                else:
                    x, feature_map = block_output, x

                if i == 0:
                    feature_maps.extend([
                        self.skip_max3_pool(feature_map)
                    ])
                else:
                    features.extend([
                        self.max_pool(feature_map)
                    ])
                
        return x, torch.cat(features, dim=1)

class Model(pytorch_model.PyTorchModel):
    def __init__(self, nb_classes, input_image=None, **kwargs):
        super(Model, self).__init__()

        self.input_channels = input_image.size()[-3]
        self.nb_classes = nb_classes

        nb_stages = 4
        lengths = [1, 2, 2, 2]
        skip_lengths = [0, 0, 0, 0]
        initial_channels = 16
        final_channels = 48

        #dropout = 0.0
        #block = ResidualBlock
        #block = lambda *a, **k: conv_block(*a, **k, groups=1, dropout=dropout)
        #block = build.block_factory(conv_block, 
        #        groups=1, dropout=dropout)
        #first_block = lambda *a, **k: nn.Sequential(
        #        nn.Conv2d(3, 3, 3, groups=3),
        #        nn.Conv2d(3, 8, 1)
        #)
        #self.stages = nn.ModuleList()
        #self.stages.append(nn.Sequential(
        #    pytorch_gconv.P4ConvZ2(self.input_channels, initial_channels//4, 3),
        #    GroupView(),
        #    nn.BatchNorm2d(initial_channels),
        #    nn.ReLU()
        #))

        #self.stages.append(build.Stage(
        #    block,
        #    self.input_channels, initial_channels, 3,
        #    length=1,
        #))

        ## 3x3 blocks, stride = 2
        #self.stages.append(build.Stage(
        #    block,
        #    *build.channel_sizing_linear(
        #        initial_channels, final_channels, 0, nb_stages), 
        #    3,
        #    stride=2,
        #    length=lengths[0],
        #    skip_length=skip_lengths[0], 
        #    channel_sizing='last'
        #))

        #self.stages.append(build.Stage(
        #    build.block_factory(DenseConvBlock, 3),
        #    *build.channel_sizing_linear(
        #        initial_channels, final_channels, 1, nb_stages), 3,
        #    pooling_block = block,
        #    stride=2,
        #    length=lengths[1],
        #    skip_length=skip_lengths[1], 
        #    channel_sizing='last'
        #))
        ##self.stages.append(GroupView())

        #self.stages.append(build.Stage(
        #    build.block_factory(DenseConvBlock, 3),
        #    *build.channel_sizing_linear(
        #        initial_channels, final_channels, 2, nb_stages), 3,
        #    pooling_block = block,
        #    stride=1,
        #    length=lengths[2],
        #    skip_length=skip_lengths[2], 
        #    channel_sizing='last'
        #))
        ##for i in range(nb_stages-1):
        ##    self.stages.append(build.Stage(
        ##        conv_block,
        ##        *build.channel_sizing_linear(
        ##            initial_channels, final_channels, i, nb_stages), 3,
        ##        length=lengths[i],
        ##    ))

        ## 3x3 block
        #self.stages.append(build.Stage(
        #    build.block_factory(DenseConvBlock, 3),
        #    *build.channel_sizing_linear(
        #        initial_channels, final_channels, nb_stages-1, nb_stages), 3,
        #    pooling_block=block,
        #    length=lengths[nb_stages-1],
        #    skip_length=skip_lengths[nb_stages-1], 
        #    channel_sizing='last'
        #))

        ##self.stages.append(nn.Sequential(
        ##    Transformer(final_channels, 2, height=2, dropout=dropout),
        ##    nn.Flatten(),
        ##    nn.Linear(final_channels*(3**2), final_channels),
        ##    nn.ReLU(),
        ##    nn.BatchNorm1d(final_channels)
        ##))

        #self.avg_pool = nn.AdaptiveAvgPool2d(3)
        #self.max_pool = nn.AdaptiveMaxPool2d(3)

        #nb_agregate_channels = sum([
        #    build.channel_sizing_linear(
        #        initial_channels, final_channels,
        #        i, nb_stages)[0]
        #    for i in range(nb_stages)
        #]) + self.input_channels
        #nb_agregate_channels *= 2 * 9

        self.conv_net = DenseNet(
                self.input_channels, final_channels, adaptive_pooling_size=1)
        nb_agregate_channels = self.conv_net.nb_agregate_channels

        final_dropout = 0.5 #0.6*(1-(1/np.log2(final_channels + nb_agregate_channels)))
        self.feature_dropout = nn.Dropout(final_dropout)
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            #nn.Dropout(0.1),
            nn.Linear(final_channels + nb_agregate_channels, final_channels),
            nn.ReLU(),
            nn.BatchNorm1d(final_channels),
            nn.Linear(final_channels, nb_classes)
        )

        self.criterion = nn.CrossEntropyLoss()

        self.optimizer = torch.optim.AdamW(self.parameters(), lr=3e-4,
            weight_decay=0.002)


        #lr = 1e-2
        #momentum = 0.9
        #self.optimizer = torch.optim.SGD(self.parameters(), 
        #    lr=lr*(1/(1+momentum)),
        #    momentum=momentum)

        self.metrics = {
            'error_rate': lambda pred, y: 1-build.accuracy(pred, y)
        }


    def forward(self, x):
        #agregates = []
        #for stage in self.stages:
        #    agregates.extend([
        #        self.avg_pool(x), 
        #        self.max_pool(x)
        #    ])
        #    x = stage(x)
        x, agregates = self.conv_net(x)

        agregates = self.flatten(agregates)
        agregates = self.feature_dropout(agregates)
        x = self.flatten(x)
        x = torch.cat([agregates, x], dim=1)

        x = self.classifier(x)

        return x



if __name__ == '__main__':
    batch_size = 32
    height = 26
    dummy_inputs = torch.randn(batch_size, 3, height, height)
    dummy_targets = torch.randint(0, 1, (batch_size,))
    dummy_nb_classes = 10

    model = Model(nb_classes=dummy_nb_classes, input_image=dummy_inputs[0])
    print(model)
    torchsummary.summary(model, input_size=dummy_inputs.size()[1:])

    dummy_outputs = model(dummy_inputs)
    if isinstance(dummy_outputs, dict):
        dummy_predictions = dummy_outputs['logits']
    else:
        dummy_predictions = dummy_outputs
    print(dummy_predictions[0])

    loss = model.criterion(dummy_outputs, dummy_targets)
    loss.backward()
    model.optimizer.step()
    print(loss.item())
