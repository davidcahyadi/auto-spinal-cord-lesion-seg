from typing import Dict, Optional

import torch
import torch.nn.init as init
from torch import nn
from torchvision.transforms.functional import center_crop

from src.model.base import Model
from src.schema import SegmentationConfig


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, use_batch_norm=True, use_xavier=True):
        super().__init__()
        self.use_batch_norm = use_batch_norm
        self._conv_1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1, bias=False)
        self._conv_2 = nn.Conv2d(
            out_channels, out_channels, 3, 1, padding=1, bias=False
        )
        self._relu = nn.ReLU(inplace=True)
        if use_batch_norm:
            self._bn_1 = nn.BatchNorm2d(out_channels)
            self._bn_2 = nn.BatchNorm2d(out_channels)

        if use_xavier:
            self.init_weights()

    def init_weights(self):
        init.xavier_uniform_(self._conv_1.weight)
        init.xavier_uniform_(self._conv_2.weight)
        if self.use_batch_norm:
            init.constant_(self._bn_1.weight, 1)
            init.constant_(self._bn_1.bias, 0)
            init.constant_(self._bn_2.weight, 1)
            init.constant_(self._bn_2.bias, 0)

    def forward(self, x):
        x = self._conv_1(x)
        if self.use_batch_norm:
            x = self._bn_1(x)
        x = self._relu(x)
        x = self._conv_2(x)
        if self.use_batch_norm:
            x = self._bn_2(x)
        x = self._relu(x)
        return x


class DownBlock(Block):
    def __init__(self, in_channels, out_channels, use_batch_norm=True, use_xavier=True):
        super().__init__(in_channels, out_channels, use_batch_norm, use_xavier)
        self._pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = super().forward(x)
        return self._pool(x), x


class UpBlock(Block):
    def __init__(self, in_channels, out_channels, use_batch_norm=True, use_xavier=True):
        super().__init__(in_channels, out_channels, use_batch_norm, use_xavier)
        self._conv_up = nn.ConvTranspose2d(
            in_channels, in_channels // 2, kernel_size=2, stride=2
        )
        if use_xavier:
            init.xavier_uniform_(self._conv_up.weight)

    def forward(self, x, skip):
        x = self._conv_up(x)
        skip = center_crop(skip, [x.shape[-2], x.shape[-1]])
        x = torch.cat([x, skip], dim=1)
        return super().forward(x)


class STAMPConfig(SegmentationConfig):
    in_channels: int = 1
    out_channels: int = 1
    depth: int = 4
    features: int = 64
    out_activation: Optional[str] = None
    use_batch_norm: bool = False
    use_xavier: bool = False
    b_drop: float = 0.1
    recovery_epochs: int = 5


class STAMPUNet(Model):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.input_layer = nn.Conv2d(
            self.hparams.in_channels,
            self.hparams.in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        if self.hparams.use_xavier:
            init.xavier_uniform_(self.input_layer.weight)

        # down part
        in_feature = self.hparams.in_channels
        out_feature = self.hparams.features
        for _ in range(self.hparams.depth):
            self.downs.append(
                DownBlock(
                    in_feature,
                    out_feature,
                    self.hparams.use_batch_norm,
                    self.hparams.use_xavier,
                )
            )
            in_feature = out_feature
            out_feature = out_feature * 2

        # middle
        self.bottleneck = Block(
            in_feature,
            out_feature,
            self.hparams.use_batch_norm,
            self.hparams.use_xavier,
        )

        # up part
        for _ in range(self.hparams.depth):
            in_feature = out_feature
            out_feature = in_feature // 2
            self.ups.append(
                UpBlock(
                    in_feature,
                    out_feature,
                    self.hparams.use_batch_norm,
                    self.hparams.use_xavier,
                )
            )

        self.output_layer = nn.Conv2d(
            self.hparams.features, self.hparams.out_channels, kernel_size=1
        )

        self.current_epoch_in_phase = 0

    def forward(self, x):
        skip_connections = []
        x = self.input_layer(x)

        for down in self.downs:
            x, x_skip = down(x)
            skip_connections.append(x_skip)

        x = self.bottleneck(x)

        for up in self.ups:
            x = up(x, skip_connections.pop())

        x = self.output_layer(x)
        x = self.out_activation(x)
        return x

    def on_train_epoch_end(self):
        self.current_epoch_in_phase += 1
        if self.current_epoch_in_phase >= self.hparams.recovery_epochs:
            self.prune_model()
            self.current_epoch_in_phase = 0
            print(self)

    def prune_model(self):
        magnitudes = self.calculate_filter_magnitudes()
        normalized_magnitudes = self.normalize_magnitudes(magnitudes)
        layer_to_prune, filter_to_prune = self.select_filter_to_prune(
            normalized_magnitudes
        )
        self.remove_filter(layer_to_prune, filter_to_prune)
        # self.update_targeted_dropout(normalized_magnitudes)
        print("Finish prune")

    def calculate_filter_magnitudes(self):
        magnitudes = {}
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                activations = module.weight.data
                magnitudes[name] = torch.norm(activations, p=2, dim=(1, 2, 3))
        print("Calculate magnitudes")
        return magnitudes

    def normalize_magnitudes(self, magnitudes):
        normalized = {}
        for name, mag in magnitudes.items():
            normalized[name] = mag / torch.sqrt(torch.sum(mag**2))

        print("Normalized magnitudes")
        return normalized

    def select_filter_to_prune(self, normalized_magnitudes):
        min_magnitude = float("inf")
        layer_to_prune = None
        filter_to_prune = None

        for name, magnitudes in normalized_magnitudes.items():
            min_mag, min_idx = torch.min(magnitudes, dim=0)
            if min_mag < min_magnitude:
                min_magnitude = min_mag
                layer_to_prune = name
                filter_to_prune = min_idx.item()

        print("Select filter to prune:", layer_to_prune, filter_to_prune)
        return layer_to_prune, filter_to_prune

    def update_targeted_dropout(self, normalized_magnitudes):
        dropout_probs = {}
        for name, magnitudes in normalized_magnitudes.items():
            avg_index = torch.argsort(magnitudes).float().mean().item()
            dropout_probs[name] = (avg_index / len(magnitudes)) * self.hparams.b_drop

        self.add_dropout_layers(dropout_probs)
        print("Finish update targeted dropout")

    def add_dropout_layers(self, dropout_probs):
        new_modules = {}
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                parent_name = ".".join(name.split(".")[:-1])
                module_name = name.split(".")[-1]
                dropout = nn.Dropout2d(p=dropout_probs.get(name, self.hparams.b_drop))
                new_modules[parent_name + "." + module_name + "_dropout"] = dropout

        for name, module in new_modules.items():
            parent_name = ".".join(name.split(".")[:-1])
            module_name = name.split(".")[-1]
            if parent_name != "":
                parent = self.get_module(parent_name)
                setattr(parent, module_name, module)
        print("Finish add dropout layers")

    def remove_filter(self, layer_name, filter_idx):
        parts = layer_name.split(".")
        module = self

        for part in parts[:-1]:
            if part.isdigit():
                module = module[int(part)]
            else:
                module = getattr(module, part)

        if isinstance(module, UpBlock):
            self.remove_filter_from_upblock(module, filter_idx)
        elif isinstance(module, Block):
            self.remove_filter_from_block(module, filter_idx)

        print("Pruned model")
        print(module)

        self.update_affected_layers(layer_name, filter_idx)

    def remove_filter_from_block(self, block: Block, filter_idx: int):
        # Update first conv layer
        block._conv_1 = self.remove_filter_from_conv(
            block._conv_1, filter_idx, out_channel=True
        )
        # Update second conv layer
        block._conv_2 = self.remove_filter_from_conv(
            block._conv_2, filter_idx, in_channel=True, out_channel=True
        )

        # Update batch norm layers if used
        if block.use_batch_norm:
            block._bn_1 = self.remove_filter_from_bn(block._bn_1, filter_idx)
            block._bn_2 = self.remove_filter_from_bn(block._bn_2, filter_idx)

    def remove_filter_from_upblock(self, upblock: UpBlock, filter_idx: int):
        # Update ConvTranspose2d layer
        upblock._conv_up = self.remove_filter_from_up_conv(
            upblock._conv_up, filter_idx, out_channel=True
        )

        # Update first conv layer
        upblock._conv_1 = self.remove_filter_from_conv(
            upblock._conv_1, filter_idx, in_channel=True, out_channel=True
        )
        # Update second conv layer
        upblock._conv_2 = self.remove_filter_from_conv(
            upblock._conv_2, filter_idx, in_channel=True, out_channel=True
        )

        # Update batch norm layers if used
        if upblock.use_batch_norm:
            upblock._bn_1 = self.remove_filter_from_bn(upblock._bn_1, filter_idx)
            upblock._bn_2 = self.remove_filter_from_bn(upblock._bn_2, filter_idx)

    def remove_affected_filter_from_block(self, block: Block, filter_idx: int):
        # Update first conv layer
        block._conv_1 = self.remove_filter_from_conv(
            block._conv_1, filter_idx, in_channel=True
        )

    def remove_affected_filter_from_upblock(self, upblock: UpBlock, filter_idx: int):
        # Update ConvTranspose2d layer
        # because transpose, the out channels and in channels are swapped
        upblock._conv_up = self.remove_filter_from_up_conv(
            upblock._conv_up, filter_idx, in_channel=True
        )

    def remove_filter_from_conv(
        self, conv: nn.Conv2d, filter_idx, in_channel=False, out_channel=False
    ):
        new_in_channels = conv.in_channels - (1 if in_channel else 0)
        new_out_channels = conv.out_channels - (1 if out_channel else 0)

        # Prevent pruning if it would result in 0 channels
        if new_in_channels == 0 or new_out_channels == 0:
            print(
                f"Warning: Cannot prune filter {filter_idx} as it would result in 0 channels."
            )
            return conv

        new_conv = nn.Conv2d(
            new_in_channels,
            new_out_channels,
            conv.kernel_size,
            conv.stride,
            conv.padding,
            conv.dilation,
            conv.groups,
            conv.bias is not None,
            conv.padding_mode,
        )

        with torch.no_grad():
            if in_channel and out_channel:
                new_weight = conv.weight.data.clone()
                new_weight = torch.cat(
                    [new_weight[:filter_idx], new_weight[filter_idx + 1 :]], dim=0
                )
                new_weight = torch.cat(
                    [new_weight[:, :filter_idx], new_weight[:, filter_idx + 1 :]], dim=1
                )
                new_conv.weight.data = new_weight
            elif in_channel:
                new_weight = conv.weight.data.clone()
                new_weight = torch.cat(
                    [new_weight[:, :filter_idx], new_weight[:, filter_idx + 1 :]], dim=1
                )
                new_conv.weight.data = new_weight
            elif out_channel:
                new_weight = conv.weight.data.clone()
                new_weight = torch.cat(
                    [new_weight[:filter_idx], new_weight[filter_idx + 1 :]], dim=0
                )
                new_conv.weight.data = new_weight

            if conv.bias is not None:
                if isinstance(conv, nn.ConvTranspose2d):
                    if out_channel:
                        new_bias = conv.bias.data.clone()
                        new_bias = torch.cat(
                            [new_bias[:filter_idx], new_bias[filter_idx + 1 :]]
                        )
                        print(new_bias.shape)
                        new_conv.bias.data = new_bias
                else:
                    if out_channel:
                        new_bias = conv.bias.data.clone()
                        new_bias = torch.cat(
                            [new_bias[:filter_idx], new_bias[filter_idx + 1 :]]
                        )
                        print(new_bias.shape)
                        new_conv.bias.data = new_bias

        device = next(conv.parameters()).device
        new_conv = new_conv.to(device)
        if self.hparams.use_xavier:
            init.xavier_uniform_(new_conv.weight)
        return new_conv

    def remove_filter_from_up_conv(
        self, conv: nn.ConvTranspose2d, filter_idx, in_channel=False, out_channel=False
    ):
        new_in_channels = conv.in_channels - (1 if in_channel else 0)
        new_out_channels = conv.out_channels - (1 if out_channel else 0)

        # Prevent pruning if it would result in 0 channels
        if new_in_channels == 0 or new_out_channels == 0:
            print(
                f"Warning: Cannot prune filter {filter_idx} as it would result in 0 channels."
            )
            return conv

        new_conv = nn.ConvTranspose2d(
            new_in_channels,
            new_out_channels,
            conv.kernel_size,
            conv.stride,
            conv.padding,
            conv.output_padding,
            conv.groups,
            conv.bias is not None,
            conv.dilation,
            conv.padding_mode,
        )

        with torch.no_grad():
            if in_channel and out_channel:
                new_weight = conv.weight.data.clone()
                new_weight = torch.cat(
                    [new_weight[:, :filter_idx], new_weight[:, filter_idx + 1 :]], dim=1
                )
                new_weight = torch.cat(
                    [new_weight[:filter_idx], new_weight[filter_idx + 1 :]], dim=0
                )
                new_conv.weight.data = new_weight
            elif in_channel:
                new_weight = conv.weight.data.clone()
                new_weight = torch.cat(
                    [new_weight[:filter_idx], new_weight[filter_idx + 1 :]], dim=0
                )
                new_conv.weight.data = new_weight
            elif out_channel:
                new_weight = conv.weight.data.clone()
                new_weight = torch.cat(
                    [new_weight[:, :filter_idx], new_weight[:, filter_idx + 1 :]], dim=1
                )
                new_conv.weight.data = new_weight

            if conv.bias is not None:
                if out_channel:
                    new_bias = conv.bias.data.clone()
                    new_bias = torch.cat(
                        [new_bias[:filter_idx], new_bias[filter_idx + 1 :]]
                    )
                    new_conv.bias.data = new_bias

        device = next(conv.parameters()).device
        new_conv = new_conv.to(device)
        if self.hparams.use_xavier:
            init.xavier_uniform_(new_conv.weight)

        return new_conv

    def remove_filter_from_bn(self, bn: nn.BatchNorm2d, filter_idx):
        new_num_features = bn.num_features - 1

        # Prevent pruning if it would result in 0 features
        if new_num_features == 0:
            print(
                f"Warning: Cannot prune filter {filter_idx} from BatchNorm as it would result in 0 features."
            )
            return bn

        new_bn = nn.BatchNorm2d(new_num_features)

        with torch.no_grad():
            # Copy parameters except for the pruned filter
            new_bn.weight.data = torch.cat(
                [bn.weight.data[:filter_idx], bn.weight.data[filter_idx + 1 :]]
            )
            new_bn.bias.data = torch.cat(
                [bn.bias.data[:filter_idx], bn.bias.data[filter_idx + 1 :]]
            )
            new_bn.running_mean = torch.cat(
                [bn.running_mean[:filter_idx], bn.running_mean[filter_idx + 1 :]]
            )
            new_bn.running_var = torch.cat(
                [bn.running_var[:filter_idx], bn.running_var[filter_idx + 1 :]]
            )

        device = next(bn.parameters()).device
        new_bn = new_bn.to(device)

        init.constant_(new_bn.weight, 1)
        init.constant_(new_bn.bias, 0)

        return new_bn

    def update_affected_layers(self, pruned_layer_name, filter_idx):
        print("Update affected layer")
        for name, module in self.named_modules():
            if self.is_affected_layer(pruned_layer_name, name):
                print("Next layer:", name)
                if isinstance(module, UpBlock):
                    self.remove_affected_filter_from_upblock(module, filter_idx)
                elif isinstance(module, Block):
                    self.remove_affected_filter_from_block(module, filter_idx)

            elif self.is_neighbour_affected_layer(pruned_layer_name, name):
                print("Across layer:", name)
                self.remove_affected_filter_from_block(module, filter_idx)

    def is_neighbour_affected_layer(self, pruned_layer_name, current_layer_name):
        pruned_parts = pruned_layer_name.split(".")
        current_parts = current_layer_name.split(".")

        if len(current_parts) == 1 or len(current_parts) > 2:
            return False

        if pruned_parts[0] == "downs" and current_parts[0] == "ups":
            if int(pruned_parts[1]) == self.hparams.depth - 1 - int(current_parts[1]):
                return True

        return False

    def is_affected_layer(self, pruned_layer_name, current_layer_name):
        pruned_parts = pruned_layer_name.split(".")
        current_parts = current_layer_name.split(".")

        if (
            len(current_parts) == 1
            and current_parts[0] != "bottleneck"
            and current_parts[0] != "output_layer"
        ):
            return False

        if pruned_parts[0] == "downs":
            # Affect next down block or up block
            if (
                current_parts[0] == "downs"
                and int(current_parts[1]) == int(pruned_parts[1]) + 1
            ):
                return True

            if (
                current_parts[0] == "bottleneck"
                and int(pruned_parts[1]) == self.hparams.depth - 1
            ):
                return True

        elif pruned_parts[0] == "bottleneck":
            if current_parts[0] == "ups" and int(current_parts[1]) == 0:
                return True

        elif pruned_parts[0] == "ups":
            # Affect next up block
            if (
                current_parts[0] == "ups"
                and int(current_parts[1]) == int(pruned_parts[1]) + 1
            ):
                return True

            if (
                current_parts[0] == "output_layer"
                and int(pruned_parts[1]) == self.hparams.depth - 1
            ):
                return True

        return False

    def print_layer_channels(self):
        print("\n--- Layer Channel Information ---")
        for name, module in self.named_modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                print(
                    f"Layer: {name} In: {module.in_channels}, Out: {module.out_channels}"
                )
                print(f"  Weight shape: {module.weight.shape}")
                if module.bias is not None:
                    print(f"  Bias shape: {module.bias.shape}")
                print()
        print("--- End of Layer Channel Information ---\n")

    def get_module(self, name):
        parts = name.split(".")
        module = self
        for part in parts:
            if part.isdigit():
                module = module[int(part)]
            else:
                module = getattr(module, part)
        return module

    def on_save_checkpoint(self, checkpoint):
        # Save the current architecture information
        checkpoint["architecture"] = self.get_architecture()

    def on_load_checkpoint(self, checkpoint):
        # Check if the checkpoint contains architecture information
        if "architecture" in checkpoint:
            # Rebuild the model architecture before loading the state dict
            self.rebuild_architecture(checkpoint["architecture"])

    def rebuild_architecture(self, architecture: Dict):
        # Clear existing modules
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        # Rebuild input layer
        in_channels = architecture["input_layer"]["in_channels"]
        out_channels = architecture["input_layer"]["out_channels"]
        self.input_layer = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        init.xavier_uniform_(self.input_layer.weight)

        # Rebuild down blocks
        for block_arch in architecture["downs"]:
            block = DownBlock(
                block_arch["in_channels"],
                block_arch["out_channels"],
                self.hparams.use_batch_norm,
            )
            self.downs.append(block)

        # Rebuild bottleneck
        bottleneck_arch = architecture["bottleneck"]
        self.bottleneck = Block(
            bottleneck_arch["in_channels"],
            bottleneck_arch["out_channels"],
            self.hparams.use_batch_norm,
        )

        # Rebuild up blocks
        for block_arch in architecture["ups"]:
            block = UpBlock(
                block_arch["in_channels"],
                block_arch["out_channels"],
                self.hparams.use_batch_norm,
            )
            block._conv_up = nn.ConvTranspose2d(
                block_arch["conv_up_in"],
                block_arch["conv_up_out"],
                kernel_size=2,
                stride=2,
            )
            init.xavier_uniform_(block._conv_up.weight)
            self.ups.append(block)

        # Rebuild output layer
        out_layer_arch = architecture["output_layer"]
        self.output_layer = nn.Conv2d(
            out_layer_arch["in_channels"], out_layer_arch["out_channels"], kernel_size=1
        )
        init.xavier_uniform_(self.output_layer.weight)

        # Rebuild activation
        if self.hparams.out_activation is None:
            self.out_activation = nn.Identity()
        elif self.hparams.out_activation.lower() == "sigmoid":
            self.out_activation = nn.Sigmoid()
        elif self.hparams.out_activation.lower() == "softmax":
            self.out_activation = nn.Softmax(dim=1)

    def get_architecture(self) -> Dict:
        architecture = {
            "input_layer": {
                "in_channels": self.input_layer.in_channels,
                "out_channels": self.input_layer.out_channels,
            },
            "downs": [],
            "bottleneck": {
                "in_channels": self.bottleneck._conv_1.in_channels,
                "out_channels": self.bottleneck._conv_2.out_channels,
            },
            "ups": [],
            "output_layer": {
                "in_channels": self.output_layer.in_channels,
                "out_channels": self.output_layer.out_channels,
            },
        }

        for down in self.downs:
            architecture["downs"].append(
                {
                    "in_channels": down._conv_1.in_channels,
                    "out_channels": down._conv_2.out_channels,
                }
            )

        for up in self.ups:
            architecture["ups"].append(
                {
                    "in_channels": up._conv_1.in_channels,
                    "out_channels": up._conv_2.out_channels,
                    "conv_up_in": up._conv_up.in_channels,
                    "conv_up_out": up._conv_up.out_channels,
                }
            )

        return architecture
