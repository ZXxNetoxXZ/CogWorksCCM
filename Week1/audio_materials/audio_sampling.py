
def turn_off_ticks(ax):
    for tic in ax.xaxis.get_major_ticks():
        tic.tick1On = tic.tick2On = False
        tic.label1On = tic.label2On = False
    for tic in ax.yaxis.get_major_ticks():
        tic.tick1On = tic.tick2On = False
        tic.label1On = tic.label2On = False

def analog_to_digital(fig, ax, sampling_rate, quantizing_bits, digital_graph = False):
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import logistic
        

    x                   = 2. # s.
    analog_time         = np.linspace (0, x, int(10e5)) # s.

    sample_number       = x * sampling_rate 
    sampling_time       = np.linspace (0, x, int(sample_number))

    quantizing_levels   = 2 ** quantizing_bits / 2
    quantizing_step     = 1. / quantizing_levels

    def analog_signal (time_point):
        return 1 / (1 + np.exp(-10*(time_point - 1)))  #logistic.cdf(time_point)


    sampling_signal     = analog_signal (sampling_time);
    quantizing_signal   = np.round (sampling_signal / quantizing_step) * quantizing_step

    if not digital_graph:
        ax.plot (analog_time, analog_signal (analog_time))
        ax.stem (sampling_time, quantizing_signal, markerfmt='r_', linefmt = ' ', basefmt = ' ')
        ax.set_title("Analog to digital signal conversion")
        ax.set_xlabel("Time")
        ax.set_ylabel("Amplitude")
    else:
        new_l = len(analog_time) / len(quantizing_signal)
        new_y = []
        for i in range(len(quantizing_signal)):
            new_y += [quantizing_signal[i]] * int(new_l)
        
        ax.plot(analog_time, analog_signal(analog_time))
        ax.plot(analog_time, new_y, alpha=0.7)
        turn_off_ticks(ax)



def song_to_digital(local_song_path = None, sampling_rate = 40, quantizing_bits = 4):
    import urllib
    import librosa
    import numpy as np
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots()
    
    if local_song_path is None:
        song_path = 'test_song.mp3'
    else: 
        song_path = local_song_path
    
    samples, fs = librosa.load(song_path, sr=44100, mono=True)
    samples = samples[:11 * 44100]
    time = np.linspace(0,11, 11 * 44100)
    
    ax.plot(time[::10], samples[::10])
    
    skip = int(len(samples) / (11 * sampling_rate))
    sampling_signal     = samples[::skip]

    quantizing_levels   = 2 ** quantizing_bits / 2
    quantizing_step     = 1. / quantizing_levels
    
    quantizing_signal   = np.round (sampling_signal / quantizing_step) * quantizing_step;

    new_l = len(time) / len(quantizing_signal)
    new_y = []
    for i in range(len(quantizing_signal)):
        new_y += [quantizing_signal[i]] * int(new_l)

    ax.plot (time, new_y);
    ax.set_title("Analog to digital signal conversion")
    ax.set_xlabel("Time(s)")
    ax.set_ylabel("Amplitude")