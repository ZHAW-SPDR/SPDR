from spdr.vad import SPDR_Vad
segment_path = './data/out/EDI_20071128-1000_ci01_NONE'
vad = SPDR_Vad()

for i, is_speech in enumerate(vad.classify_segments(segment_path)):
    if not is_speech:
        print("segment %d isn't speech" % i)
