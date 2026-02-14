def segment_signal(signal, segment_length):
    segments = []
    for i in range(0, len(signal) - segment_length, segment_length):
        segments.append(signal[i:i+segment_length])
    return segments
