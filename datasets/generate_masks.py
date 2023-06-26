#%%
from psychopy import visual, core

#%%
# Stimuli are 976 x 1116
# width = 976
height = 1116
width = height
for i in range(10):
    win = visual.Window(size=(width, height))
    noise = visual.NoiseStim(win, name='noise',units='pix',
                    mask=None,
                    ori=0.0, pos=(0, 0), size=(width, height), sf=None, phase=0,
                    color=[1,1,1], colorSpace='rgb', opacity=1, blendmode='add', contrast=1.0,
                    filter='butterworth', imageComponent='Phase',
                    noiseType='Binary', noiseElementSize=20, 
                    noiseBW=1.0, noiseBWO=30, noiseFractalPower=-1, noiseFilterLower=7/1116, noiseFilterUpper=12/1116,
                    noiseFilterOrder=3.0, noiseClip=1.0, interpolate=False, depth=-1.0)
    # noise = visual.NoiseStim(win,
    #                 noiseImage='testImg.jpg', mask=None,
    #                 size=(42*10, 48*10),
    #                 noiseType='Binary', noiseElementSize=5)
    noise.draw()
    win.flip()
    # Each Frame
    win.getMovieFrame('front')

    # End Routine
    fname = f'structured_noise_{str(i).zfill(2)}.jpg'
    print(f'Saving {fname}')
    win.saveMovieFrames(fname)
    win.close()
# %%
win.close()
# %%

# 3/480 = 0.00625 
# 6/480 = 0.0125
# 1116/480 = 2.325
# 7/1116 = 0.00627240143
# 14/1116 = 0.01254480286
