# @title Downloading the Indic Models
import torch,glob
from maha_tts import load_models,infer_tts,config
from scipy.io.wavfile import write
from IPython.display import Audio,display

# PATH TO THE SPEAKERS WAV FILES
speaker =['/content/infer_ref_wavs/2272_152282_000019_000001/',
          '/content/infer_ref_wavs/2971_4275_000049_000000/',
          '/content/infer_ref_wavs/4807_26852_000062_000000/',
          '/content/infer_ref_wavs/6518_66470_000014_000002/']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
diff_model,ts_model,vocoder,diffuser = load_models('Smolie-in',device)
print('Using:',device)

# @title Generate for every speaker
text = "हसो मुस्कुराओ और मज़ा करो!, इस नए साल खुशियों का स्वागत करो और दुखों को फेंको बाहर!" # @param {type:"string"}
language = 'hindi' # @param {type:"string"}
language = torch.tensor(config.lang_index[language]).to(device).unsqueeze(0)
for i in speaker:
  ref_clips = glob.glob(i+'*.wav')
  audio,sr = infer_tts(text,ref_clips,diffuser,diff_model,ts_model,vocoder,language)

  write('/content/test_%s.wav'%(i.split('/')[-2]),sr,audio)
  print(text)
  display(Audio('/content/test_%s.wav'%(i.split('/')[-2])))