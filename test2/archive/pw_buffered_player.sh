#!/bin/bash
# Buffered Bluetooth audio player using PipeWire tools
# Captures from bluez_input, buffers 10 seconds, then plays

BUFFER_SEC=${1:-10}
RATE=44100
BUFFER_BYTES=$((BUFFER_SEC * RATE * 2 * 2))  # stereo 16-bit

echo "Buffered BT Player - ${BUFFER_SEC}s buffer"
echo "Capturing from Bluetooth, buffering, then playing..."

# Find the bluez input node by name
BT_NODE=$(pw-cli ls Node 2>/dev/null | grep -A1 "bluez_input" | grep node.name | awk -F'"' '{print $2}')

if [ -z "$BT_NODE" ]; then
    echo "No Bluetooth input found!"
    exit 1
fi

echo "Found BT node: $BT_NODE"
echo "Buffering ${BUFFER_SEC} seconds before playback..."

# Use a FIFO for buffering
FIFO=/tmp/bt_audio_buffer
rm -f $FIFO
mkfifo $FIFO

# Start capture in background, writing to FIFO
pw-cat --target "$BT_NODE" -r --format s16 --rate $RATE --channels 2 - > $FIFO 2>/dev/null &
CAPTURE_PID=$!

# Wait for buffer to fill
echo "Filling buffer..."
dd if=$FIFO bs=$BUFFER_BYTES count=1 of=/tmp/bt_prebuffer.raw 2>/dev/null

echo "Buffer full - starting playback"

# Play prebuffer then continue from FIFO
(cat /tmp/bt_prebuffer.raw; cat $FIFO) | pw-play --target alsa_output.usb-Pollen_Robotics_Reachy_Mini_Audio_202000386253800122-00.playback.0.0 --format s16 --rate $RATE --channels 2 - &
PLAY_PID=$!

echo "Playing... Press Ctrl+C to stop"

cleanup() {
    echo "Stopping..."
    kill $CAPTURE_PID $PLAY_PID 2>/dev/null
    rm -f $FIFO /tmp/bt_prebuffer.raw
    exit 0
}

trap cleanup SIGINT SIGTERM

wait $PLAY_PID
