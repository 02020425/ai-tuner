import numpy as np, soundfile as sf, os
os.makedirs('data/clean', exist_ok=True)
sr = 22050
for i in range(10):
    dur = np.random.uniform(2, 4)
    t = np.linspace(0, dur, int(sr*dur), endpoint=False)
    f0 = np.random.uniform(200, 500)
    vibrato = 0.003 * np.sin(2*np.pi*5*t) * np.sin(2*np.pi*f0*t)
    y = 0.5 * np.sin(2*np.pi*f0*t + vibrato) * np.exp(-0.3*t)
    sf.write(f'data/clean/synth_{i:02d}.wav', y.astype(np.float32), sr)
print('Generated', len(os.listdir('data/clean')), 'files')
