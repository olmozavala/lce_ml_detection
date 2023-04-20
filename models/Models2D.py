import torch.nn as nn
import torch
from collections import OrderedDict

class EncoderDecoderBlock(nn.Module):
    def __init__(self, level, cnn_per_level=2, activation=nn.ReLU, in_filters=8, out_filters=16, kernel_size=3,
                 add_transpose=False, batch_norm=True):
        super().__init__()
        block_layers = OrderedDict()
        for i in range(cnn_per_level):
            layer_name = f'conv_l{level}_{i}'
            if i == 0:
                block_layers[layer_name] = nn.Conv2d(in_filters, out_filters, kernel_size, 1, padding='same')
            else:
                block_layers[layer_name] = nn.Conv2d(out_filters, out_filters, kernel_size, 1, padding='same')

            if batch_norm:
                layer_name = f'batch_l{level}_{i}'
                block_layers[layer_name] = nn.BatchNorm2d(out_filters)

            layer_name = f'act_l{level}_{i}'
            block_layers[layer_name] = activation()

        if add_transpose:
            layer_name = f'transpose_l{level}'
            block_layers[layer_name] = nn.ConvTranspose2d(out_filters, out_filters, 2, 2)
        self.block = nn.Sequential(block_layers)

    def forward(self, x):
        return self.block(x)

# Define the CNN architecture
class UNet(nn.Module):

    def __init__(self, num_levels=4, cnn_per_level=2,  input_channels=1, output_channels=1, start_filters=16,
                 kernel_size=3, out_activation=nn.Sigmoid, batch_norm=True):
        '''
        :param num_levels:
        :param cnn_per_level:
        :param input_channels:
        :param output_channels:
        :param start_filters:
        :param kernel_size:
        '''
        super(UNet, self).__init__()
        print("UNet model")
        cur_filters = -1  # Just initialize current filters
        encoder_blocks = OrderedDict()
        decoder_blocks = OrderedDict()
        self.maxpools = nn.ModuleList()
        self.levels = num_levels
        self.maxpool = nn.MaxPool2d(2, 2)
        # Simulated inputs (just to see how the dimensions get modified)
        cur_w = 136
        cur_h = 232
        print("---------- Encoder ----------")
        for c_level in range(1, num_levels):
            input_filters = input_channels if c_level == 1 else output_filters
            output_filters = start_filters if c_level == 1 else output_filters * 2
            print(f'Level {c_level} in: {input_filters}x{cur_w}x{cur_h}')
            c_encoder = EncoderDecoderBlock(c_level, cnn_per_level, nn.ReLU, in_filters=input_filters,
                                            out_filters=output_filters, kernel_size=kernel_size, batch_norm=batch_norm)
            encoder_blocks[f'enc_lev_{c_level}'] = c_encoder
            cur_w = int((cur_w)/ 2)    # for padding = same
            cur_h = int((cur_h)/ 2)    # for padding = same
            print(f'-- Level {c_level} out: {output_filters}x{cur_w}x{cur_h}')
            cur_filters = output_filters
        # Decoder
        self.encoder_blocks = nn.Sequential(encoder_blocks)
        self.maxpools = nn.ModuleList([nn.MaxPool2d(2, 2) for _ in range(num_levels-1)])

        print("---------- Bottom ----------")
        input_filters = cur_filters   # Considering the skip connections
        print(f'-- Level Bottom in: {input_filters}x{cur_w}x{cur_h}')
        output_filters = int(cur_filters * 2)  # Considering the skip connections
        self.bottom = EncoderDecoderBlock('bottom', cnn_per_level, nn.ReLU, in_filters=input_filters,
                                          out_filters=output_filters, kernel_size=kernel_size, batch_norm=batch_norm,
                                          add_transpose=True)
        cur_w = int((cur_w) * 2)  # for padding = same
        cur_h = int((cur_h) * 2)  # for padding = same
        print(f'-- Level Bottom out: {output_filters}x{cur_w}x{cur_h}')
        cur_filters = output_filters

        print("---------- Decoder ----------")
        for c_level in range(num_levels-1, 0, -1):
            input_filters = cur_filters + int(cur_filters/2)  # Considering the skip connections
            print(f'Level {c_level} in: {input_filters}(skip)x{cur_w}x{cur_h}')

            add_transpose = False if c_level == 1 else True
            output_filters = int(cur_filters/2) if add_transpose else input_filters

            decoder_blocks[f'dec_lev_{c_level}'] = EncoderDecoderBlock(c_level, cnn_per_level, nn.ReLU, input_filters, output_filters,
                                                                       kernel_size, add_transpose=add_transpose, batch_norm=batch_norm)
            # cur_w = int((cur_w - 4) * 2) if add_transpose else cur_w - 4  # For padding = valid
            cur_w = int((cur_w) * 2) if add_transpose else cur_w  # For padding = same
            cur_h = int((cur_h) * 2) if add_transpose else cur_h  # For padding = same
            print(f'-- Level {c_level} out: {output_filters}x{cur_w}x{cur_h}')
            cur_filters = output_filters

        self.decoder_blocks = nn.Sequential(decoder_blocks)
        print(f'* Last layer in: {output_filters}x{cur_w}x{cur_h}')
        print(f'* Last layer out: {output_channels}x{cur_w}x{cur_h}')
        self.out_layer = EncoderDecoderBlock(c_level, 1, out_activation, output_filters, output_channels, kernel_size,
                                             add_transpose=add_transpose, batch_norm=batch_norm)

    def forward(self, x):
        encoder_outputs = {}
        for level, enc_block in enumerate(self.encoder_blocks):
            # print(f"Evaluating enc_{level+1}")
            # print(x.shape)
            x = enc_block(x)
            encoder_outputs[f'enc_{level+1}'] = x
            x = self.maxpools[level](x)
            # print(x.shape)

        # print(" ------- Bottom -------")
        # print(x.shape)
        x = self.bottom(x)
        # print(x.shape)

        # print(" ------- Decoder -------")
        for dec_level, dec_block in enumerate(self.decoder_blocks):
            # print(f"Evaluating dec_{dec_level+1}")
            dec_level_str = f'enc_{self.levels - dec_level - 1}'
            # print(f"Skip connection with {dec_level_str}")
            x = torch.cat((encoder_outputs[dec_level_str], x), dim=1)
            # print(x.shape)
            x = dec_block(x)
            # print(x.shape)

        return self.out_layer(x)