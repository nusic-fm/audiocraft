import torchaudio
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

model = MusicGen.get_pretrained('facebook/musicgen-melody')
model.set_generation_params(duration=0.5)  # generate 0.25 second.
descriptions = ['sad jazz']
# wav = model.generate(descriptions)  # generates 1 audio sample with the given descriptions.

melody, sr = torchaudio.load('./assets/mourge.wav')

# encodes the description and melody for later use
encoded_conditioning = model.encode_with_chroma(descriptions, melody[None].expand(1, -1, -1), sr, "mourge_encoded")
print(encoded_conditioning)

# generates using the melody from the given audio and the provided descriptions.
# wav = model.generate_with_chroma(descriptions, melody[None].expand(1, -1, -1), sr, progress=True)



# for idx, one_wav in enumerate(wav):
    # Will save under {idx}.wav, with loudness normalization at -14 db LUFS.
    # audio_write(f'{idx}', one_wav.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)