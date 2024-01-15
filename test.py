import torchaudio
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
from torch import allclose # for testing
import os

model = MusicGen.get_pretrained('facebook/musicgen-melody')
model.set_generation_params(duration=1)  # generate 1second.
# descriptions = ['sad jazz']
# wav = model.generate(descriptions)  # generates 1 audio sample without melody with the given descriptions.

filename = "a_duck_quacking_as_birds_chirp_and_a_pigeon_cooing.mp3"

# check if encoding file exists, if not, load wav and encode
if not os.path.exists(f"{filename}_encoded.pkl") and not os.path.exists(f"{filename}_attributes.pkl"):
    print("pre-saved encoding not found, encoding now...")
    melody, sr = torchaudio.load(f"./assets/{filename}")
    descriptions = ['sad jazz']
    encoded_conditions, attributes = model.encode_with_chroma(descriptions, melody[None].expand(1, -1, -1), sr, filename)
else:
    print("pre-saved encoding found, loading...")
    encoded_conditions = model.load_encoded_conditions(f"{filename}_encoded")
    attributes = model.load_attributes(f"{filename}_attributes")

## i/o test
# loads the encoded conditioning
# encoded_conditioning_ = model.load_encoded_conditions("mourge_encoded")
# print("descriptions matching: ", allclose(encoded_conditioning['description'][0], encoded_conditioning['description'][0]))
# attributes_ = model.load_attributes("mourge_attributes")
# print("self_wavs matching: ", allclose(encoded_conditioning['self_wav'][0], encoded_conditioning['self_wav'][0]))

# generates using the encoded conditioning
wav = model.generate_with_encoded_conditions(encoded_conditions, attributes, progress=True)

# generates using the melody from the given audio and the provided descriptions.
# wav = model.generate_with_chroma(descriptions, melody[None].expand(1, -1, -1), sr, progress=True)

for idx, one_wav in enumerate(wav):
    # Will save under {idx}.wav, with loudness normalization at -14 db LUFS.
    audio_write(f'{idx}', one_wav.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)