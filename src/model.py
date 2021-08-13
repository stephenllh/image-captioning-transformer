import torch
from torch import nn
import timm


class ImageCaptioningModel(nn.Module):
    def __init__(
        self,
        cnn_backbone,
        image_size,
        vocab_size,
        max_seq_length,
        num_decoder_layers,
        n_head,
        d_model,
        fc_dim,
        dropout,
    ):
        super().__init__()

        # Image
        self.encoder = timm.create_model(cnn_backbone, pretrained=True)
        for param in self.encoder.parameters():
            self.encoder.requires_grad = False

        # Determine the input channel of the linear layer
        dummy_input = torch.randn((1, 3, image_size, image_size))
        with torch.no_grad():
            linear_in_channels = self.encoder(dummy_input).shape[1]
        self.encoder_linear = nn.Linear(linear_in_channels, d_model)

        # Text
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_embedding = nn.Embedding(max_seq_length, d_model)
        # decoder_layer = MyTransformerDecoderLayer(d_model, n_head, fc_dim, dropout, batch_first=True)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model, n_head, fc_dim, dropout, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        self.linear = nn.Linear(d_model, vocab_size)
        self.max_seq_length = max_seq_length
        self.d_model = d_model

    def forward(self, image, caption, target_pad_mask):
        batch_size, sequence_length = caption.shape[0], caption.shape[1]

        # Positional embedding at the decoder
        scale = torch.sqrt(torch.tensor([self.d_model])).to(image.device)
        x = self.token_embedding(caption) * scale
        position = (
            torch.arange(0, sequence_length)
            .unsqueeze(0)
            .repeat(batch_size, 1)
            .to(image.device)
        )
        x += self.positional_embedding(position)

        # Image feature at the encoder
        with torch.no_grad():
            image_feature = self.encoder(image)
        image_feature = image_feature.view(image_feature.shape[0], -1)

        # Decoder
        encoder_memory = self.encoder_linear(image_feature)
        encoder_memory = encoder_memory.unsqueeze(1)  # .permute(1, 0, 2)
        target_subsequent_mask = (
            nn.Transformer()
            .generate_square_subsequent_mask(x.shape[1])
            .to(image.device)
        )
        x = self.decoder(
            x,
            encoder_memory,
            tgt_mask=target_subsequent_mask,
            tgt_key_padding_mask=target_pad_mask,
        )
        out = self.linear(x)
        return out, None
