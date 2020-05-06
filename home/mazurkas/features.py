import footprint.tokenizers as tokenizers
from footprint.features import ocmi
#from footprint.features import crema as cremalib
import numpy as np
import librosa
from sklearn.preprocessing import normalize
from . import simplefast

def matrix_profile(feature_name, multipl):
  def get_matrix_profile(audio):
    feature = audio.features[feature_name]
    y, sr = audio.signal()
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, trim=False)
    seconds = len(y)/sr
    samples_per_beat = (60 * sr) / tempo
    feature_len = len(feature.T)
    feature_samples_per_beat = samples_per_beat * (feature_len/len(y))
    m = multipl
    sz = int(m * feature_samples_per_beat)
    sz = min(sz, feature_len/16)
    mp = simplefast.simpleself(feature, sz)[0]
    return mp
  return get_matrix_profile

def proc_ocmi_feature(feature_name, method, reshape=(None, None)):
  def feat_method(audio):
    feature = audio.features[feature_name]
    f2 = method(feature)
    return tokenizers.reshape(f2, reshape[0], reshape[1])
  return feat_method

def tempo(audio):
  print('extracting tempo for ', audio.filename)
  y, sr = audio.signal()
  tempo, beats = librosa.beat.beat_track(y=y, sr=sr, trim=False)
  return [tempo]

def beats(audio):
  print('extracting tempo for ', audio.filename)
  y, sr = audio.signal()
  tempo, beats = librosa.beat.beat_track(y=y, sr=sr, trim=False)
  return beats

def chroma_cens(audio):
  y, sr = audio.signal()
  print('running chroma for ', audio.filename)
  return librosa.feature.chroma_cens(y, sr, hop_length=2**11)

def chroma_ocmi(audio):
  chroma = audio.features['chroma_cens']
  return ocmi.ocmi(chroma)

def feat_mfcc(audio):
  #import code; code.interact(local=dict(globals(), **locals()))
  mfcc = librosa.feature.mfcc(y=audio.y, sr=audio.sr)
  return normalize(mfcc, axis=1, norm='max')

def feat_mfcc_delta(audio):
  mfcc = audio.features['mfcc']
  return librosa.feature.delta(mfcc)

def crema(audio):
  return cremalib.process(audio)

def crema_ocmi(audio):
  crema = audio.features['crema']
  return ocmi.ocmi(crema)

def beat_chroma_ocmi(audio):
  chroma = audio.features['beat_chroma_cens']
  return ocmi.ocmi(chroma)

def beat_chroma_cens(audio):
  chroma = audio.features['chroma_cens']
  beats = audio.features['beats']
  beat_f = librosa.util.fix_frames(beats, x_max=chroma.shape[1])
  return librosa.util.sync(chroma, beat_f, aggregate=np.median)
