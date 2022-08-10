import torchaudio
import torchvision

def get_spectrogram(audio_file_path, dimension=160):
    """
    Read sound file and return corresponding mel spectrogram.

    Args:
        audio_file_path (str): path of sound file
        dimension (int, defaults=160): dimension of spectrogram image

    Returns:
        torch.Tensor: spectrogram of a sound wave
    """
    # read file
    data_waveform, rate_of_sample = torchaudio.load(audio_file_path)
    
    # use VAD to clean the silence 
    cleaned_wave = torchaudio.functional.vad(data_waveform, rate_of_sample)
    
    # calculate spectrogram
    desired_shape = int(2 * cleaned_wave.shape[1] / dimension + 1)
    spectrogram = torchaudio.transforms.MelSpectrogram(n_mels=dimension, n_fft = desired_shape)(cleaned_wave)
    
    return spectrogram

def save_spectrogram_as_image(spectrogram, file_name=None):
    """
    Convert spectrogram to an image

    Args:
        spectrogram (torch.Tensor): spectrogram of a sound wave
        file_name (str, defaults=None): path to save image

    Returns:
        PIL.Image.Image: image format of spectrogram
    """
    img = torchvision.transforms.ToPILImage()(spectrogram)
    if file_name is not None:
        img.save(file_name)
    return img