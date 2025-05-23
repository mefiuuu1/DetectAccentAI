import torch
import torchaudio
from speechbrain.inference.interfaces import Pretrained
from speechbrain.processing.speech_augmentation import SpecAugment
from speechbrain.pretrained import EncoderClassifier

class CustomEncoderWav2vec2Classifier(EncoderClassifier):
    """A custom interface for the accent classification model."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.specaug = SpecAugment(
            time_warp=True,
            time_warp_window=5,
            freq_mask=True,
            n_freq_mask=2,
            time_mask=True,
            n_time_mask=2,
            replace_with_zero=True,
        )

    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        feats = self.mods.compute_features(wavs)
        feats = self.mods.mean_var_norm(feats, wav_lens)
        embeddings = self.mods.embedding_model(feats, wav_lens)
        outputs = self.mods.classifier(embeddings)
        return outputs

    def classify_file(self, path):
        """Classify a file and return probabilities, score, index, and label."""
        out_prob = self.predict_file(path)
        score, index = torch.max(out_prob, dim=1)
        text_lab = self.hparams.label_encoder.decode_torch(index)
        return out_prob.squeeze().tolist(), float(score), int(index), text_lab 