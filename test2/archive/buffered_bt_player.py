#!/usr/bin/env python3
import pyaudio
import time
import collections
import sys

# --- SETTINGS ---
BUFFER_SECONDS = 10    # The size of the "tank"
SAMPLE_RATE = 44100    # Standard Bluetooth quality
CHUNK_SIZE = 4096      # How big the buckets are

def main():
    p = pyaudio.PyAudio()

    # 1. FIND DEVICES
    bt_idx = None
    reachy_idx = None

    print("--- Scanning Audio Devices ---")
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        name = info.get('name', '')

        # Find iPhone/Bluetooth Input
        if 'bluez' in name or 'a2dp' in name.lower():
            if info['maxInputChannels'] > 0:
                print(f"  [INPUT]  ID {i}: {name}")
                bt_idx = i

        # Find Reachy Speaker Output
        if 'Reachy' in name and 'Audio' in name:
            if info['maxOutputChannels'] > 0:
                print(f"  [OUTPUT] ID {i}: {name}")
                reachy_idx = i

    if bt_idx is None:
        print("\n[ERROR] Could not find Bluetooth Input. Is iPhone connected and playing music?")
        return
    if reachy_idx is None:
        print("\n[ERROR] Could not find Reachy Speaker.")
        return

    print(f"\n[OK] Configuration: Bluetooth (ID {bt_idx}) -> 10s Buffer -> Reachy (ID {reachy_idx})")

    # 2. SETUP STREAMS
    # The Buffer "Tank"
    audio_buffer = collections.deque()
    buffer_limit = int((SAMPLE_RATE * BUFFER_SECONDS) / CHUNK_SIZE)
    is_playing = False

    def callback(in_data, frame_count, time_info, status):
        audio_buffer.append(in_data)
        return (None, pyaudio.paContinue)

    # Input Stream (Capture)
    in_stream = p.open(format=pyaudio.paInt16, channels=2, rate=SAMPLE_RATE,
                      input=True, input_device_index=bt_idx,
                      frames_per_buffer=CHUNK_SIZE, stream_callback=callback)

    # Output Stream (Playback)
    out_stream = p.open(format=pyaudio.paInt16, channels=2, rate=SAMPLE_RATE,
                       output=True, output_device_index=reachy_idx,
                       frames_per_buffer=CHUNK_SIZE)

    in_stream.start_stream()

    # 3. MAIN LOOP
    print(f"\nFilling Buffer... (Wait {BUFFER_SECONDS}s)")

    try:
        while in_stream.is_active():
            # If we aren't playing yet, just watch the buffer fill
            if not is_playing:
                current_fill = len(audio_buffer)
                percentage = int((current_fill / buffer_limit) * 100)
                sys.stdout.write(f"\rBuffering: {percentage}%")
                sys.stdout.flush()

                if current_fill >= buffer_limit:
                    print("\n\n>>> Buffer Full! STARTING PLAYBACK.")
                    is_playing = True
                time.sleep(0.1)

            # If we are playing, move data from buffer to speaker
            else:
                if len(audio_buffer) > 0:
                    data = audio_buffer.popleft()
                    out_stream.write(data) # This sends audio to the speaker
                else:
                    print("\n\n[WARN] Buffer ran dry! Pausing to refill...")
                    is_playing = False

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        p.terminate()

if __name__ == "__main__":
    main()
