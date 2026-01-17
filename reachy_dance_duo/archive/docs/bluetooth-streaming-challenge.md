# Bluetooth Streaming Challenge

## Overview

The `bluetooth_streamer` mode in Reachy Dance Suite enables real-time audio-reactive robot movement by streaming music from a phone via Bluetooth A2DP. This document describes the technical challenges encountered when porting from macOS to Raspberry Pi and the solutions implemented.

## Architecture

### Audio Flow

```
iPhone (Bluetooth A2DP Source)
    │
    ▼ Bluetooth (SBC/AAC codec, 44.1kHz)
    │
Raspberry Pi (A2DP Sink)
    │
    ▼ PipeWire/WirePlumber
    │
Reachy Mini Audio (USB Speaker, 16kHz)
    │
    ▼ Monitor Source (loopback capture)
    │
Dance App (FFT → Movement)
    │
    ▼ SafetyMixer
    │
Robot Hardware
```

### Original Design (macOS)

The app was originally designed for macOS using **BlackHole**, a virtual audio loopback driver that captures system audio. The flow was:

```
Mac System Audio → BlackHole → Dance App
```

This allowed capturing any audio playing on the Mac (Spotify, browser, etc.) with minimal latency.

## Problems Discovered

### 1. Device Pattern Mismatch

**Symptom**: Mode fails to start, no audio captured

The code had a hardcoded macOS-specific device pattern:

```python
DEVICE_PATTERNS = {
    "bluetooth_streamer": "BlackHole",  # Doesn't exist on Linux!
}
```

BlackHole is macOS-only. On Linux/Raspberry Pi, we need to use PulseAudio/PipeWire sources.

### 2. Bluetooth Transport Volume at Zero

**Symptom**: Connection appears successful, but no audio flows

The BlueZ Bluetooth stack initializes the A2DP transport volume to 0 by default. This is a D-Bus property that must be explicitly set:

```bash
# Check volume (was 0)
busctl get-property org.bluez /org/bluez/hci0/dev_XX_XX_XX_XX_XX_XX/sep2/fdN \
    org.bluez.MediaTransport1 Volume

# Set volume to max (127)
busctl set-property org.bluez /org/bluez/hci0/dev_XX_XX_XX_XX_XX_XX/sep2/fdN \
    org.bluez.MediaTransport1 Volume q 127
```

### 3. Excessive Latency Configuration

**Symptom**: Garbled, stuttering, or severely delayed audio

The WirePlumber Bluetooth configuration had extreme latency values:

```lua
-- BEFORE (broken)
["api.bluez5.a2dp.latency.msec"] = 5000,  -- 5 SECONDS!
["node.latency"] = "80000/16000",          -- Also 5 seconds
```

This caused massive audio buffering, making real-time dance response impossible.

### 4. Sample Rate Confusion

**Symptom**: Potential quality loss or playback issues

Multiple sample rates were in play:
- Bluetooth A2DP: 44.1kHz (standard)
- Reachy Mini Audio hardware: 16kHz (fixed)
- App configuration: 16kHz (to match hardware)

PipeWire handles resampling automatically, but mismatched configurations can cause issues.

## Solutions Implemented

### 1. Platform-Aware Device Patterns

Updated `audio_stream.py` to use PulseAudio on Linux:

```python
DEVICE_PATTERNS = {
    "live_groove": "Reachy Mini Audio",
    "bluetooth_streamer": "pulse",  # Uses PulseAudio default source
    "bluetooth_streamer_mac": "BlackHole",
}
```

The default PulseAudio source is set to the monitor (loopback) which captures Bluetooth audio routed to the speaker:

```bash
pactl set-default-source alsa_output.usb-Pollen_Robotics_Reachy_Mini_Audio_*.monitor
```

### 2. WirePlumber Bluetooth Configuration

Updated `~/.config/wireplumber/bluetooth.lua.d/51-bluetooth-config.lua`:

```lua
-- Optimized for real-time dance
bluez_monitor.properties = {
  ["bluez5.enable-sbc-xq"] = true,
  ["bluez5.enable-msbc"] = true,
  ["bluez5.enable-hw-volume"] = true,
  ["bluez5.codecs"] = "[ sbc sbc_xq aac ]",
  ["bluez5.roles"] = "[ a2dp_sink a2dp_source ]",
  ["bluez5.default.rate"] = 44100,
  ["bluez5.default.channels"] = 2,
}

bluez_monitor.rules = {
  -- Device-level settings
  {
    matches = {{ { "device.name", "matches", "bluez_card.*" } }},
    apply_properties = {
      ["bluez5.auto-connect"] = "[ a2dp_sink a2dp_source ]",
      ["api.bluez5.a2dp.latency.msec"] = 50,  -- 50ms, not 5000ms!
      ["session.suspend-timeout-seconds"] = 0,
    },
  },
  -- Node-level settings for Bluetooth input
  {
    matches = {{ { "node.name", "matches", "bluez_input.*" } }},
    apply_properties = {
      ["node.latency"] = "2048/44100",  -- ~46ms buffer
      ["resample.quality"] = 10,         -- High quality resampling
      ["api.bluez5.transport-volume"] = 100,  -- Auto-set volume
    },
  },
}
```

### 3. Audio Configuration

The app configuration in `config.py` remains at 16kHz to match the hardware:

```python
AUDIO_CONFIG = {
    "sample_rate": 16000,
    "chunk_size": 512,  # ~32ms latency
}
```

PipeWire automatically resamples from 44.1kHz (Bluetooth) to 16kHz (hardware).

## Connecting a Phone

### Initial Setup

1. Make the Pi discoverable:
   ```bash
   bluetoothctl discoverable on
   bluetoothctl discoverable-timeout 300
   ```

2. Trust the device for auto-connect:
   ```bash
   bluetoothctl trust XX:XX:XX:XX:XX:XX
   ```

3. Connect from the phone's Bluetooth settings

### Selecting Audio Output (iPhone)

After Bluetooth pairing, you must select the Pi as the audio output:

1. Open **Control Center** (swipe down from top-right)
2. **Long-press** the music/audio controls
3. Tap the **AirPlay/speaker icon**
4. Select the Pi's name (e.g., "zach")

### Verifying Audio Flow

```bash
# Check Bluetooth device is connected
bluetoothctl info XX:XX:XX:XX:XX:XX | grep Connected

# Check audio streams
wpctl status | grep -A10 "Streams:"

# Check transport volume (should be > 0)
busctl get-property org.bluez /org/bluez/hci0/dev_XX_XX_XX_XX_XX_XX/sep2/fdN \
    org.bluez.MediaTransport1 Volume

# Test audio capture
parecord --device=alsa_output.*.monitor -r 16000 test.wav
```

## Latency Budget

| Stage | Latency |
|-------|---------|
| Bluetooth A2DP encoding | ~20-40ms |
| Bluetooth transmission | ~20-40ms |
| PipeWire buffer | ~46ms |
| App audio buffer | ~32ms |
| FFT processing | ~1ms |
| Robot command | ~10ms |
| **Total** | **~130-170ms** |

This is acceptable for dance-reactive movement, though noticeable compared to wired audio.

## Troubleshooting

### No Audio Captured

1. Check Bluetooth connection: `bluetoothctl info XX:XX:XX:XX:XX:XX`
2. Check transport volume is not 0: `busctl get-property ... Volume`
3. Verify default source is monitor: `pactl get-default-source`
4. Ensure music is playing on the phone

### Garbled/Stuttering Audio

1. Check latency config: `cat ~/.config/wireplumber/bluetooth.lua.d/*.lua`
2. Restart WirePlumber: `systemctl --user restart wireplumber`
3. Check for buffer overruns in logs: `journalctl --user -u pipewire`

### Connection Drops

1. Check signal strength (keep phone within ~5m)
2. Disable WiFi on 2.4GHz if interference suspected
3. Check `journalctl -u bluetooth` for errors

## Hardware Limitations

- **Reachy Mini Audio**: Fixed at 16kHz sample rate
- **Raspberry Pi Bluetooth**: May have higher latency than dedicated BT adapters
- **A2DP Sink**: Pi must act as a "speaker" to receive audio; most phones expect this

## Future Improvements

1. **Direct Bluetooth capture**: Expose `bluez_input` as a PipeWire source to bypass the speaker/monitor path
2. **Codec optimization**: Prefer AAC or aptX over SBC for better quality
3. **Latency reduction**: Investigate lower buffer sizes with stable operation
4. **Auto-reconnect**: Script to automatically reconnect known devices on boot
