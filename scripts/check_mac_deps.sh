#!/bin/bash

# check_mac_deps.sh
# Verifies and installs system dependencies for Reachy Ultra Dance Mix 9000 on macOS.
# Updates logic for finding local dependencies.

set -e

# Ensure we can find brew on Apple Silicon and Intel Macs
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"

echo "🔍 Checking macOS dependencies..."

# 1. Check for Homebrew
if ! command -v brew &> /dev/null; then
    echo "❌ Homebrew not found. Please install headers first:"
    echo '/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
    exit 1
else
    echo "✅ Homebrew found."
fi

# 2. Check for PortAudio (Required for PyAudio)
if ! brew list portaudio &> /dev/null; then
    echo "⚠️  PortAudio not found. Installing..."
    brew install portaudio
else
    echo "✅ PortAudio found."
fi

# 3. Check for BlackHole 2ch (Required for Disco Diva)
# brew list --cask blackhole-2ch sometimes returns error if not found, or might be installed manually
if ! brew list --cask blackhole-2ch &> /dev/null; then
    # Check if header exists in /Library/Audio/Plug-Ins/HAL as backup check
    if [ -d "/Library/Audio/Plug-Ins/HAL/BlackHole2ch.driver" ]; then
        echo "✅ BlackHole 2ch driver found (manual install?)."
    else
        echo "⚠️  BlackHole 2ch not found. Installing..."
        brew install --cask blackhole-2ch
        echo "ℹ️  Note: You may need to grant Microphone permissions to Terminal/Python if requested."
    fi
else
    echo "✅ BlackHole 2ch found."
fi

# 4. Optional: FFmpeg (Good to have, though we migrated ConnectedChoreographer to PyAV)
if ! brew list ffmpeg &> /dev/null; then
    echo "ℹ️  FFmpeg not found. It is recommended but not strictly required for the latest version."
    read -p "   Install FFmpeg? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        brew install ffmpeg
    fi
else
    echo "✅ FFmpeg found."
fi

# 5. Check specific Python version (3.10+) logic is usually handled by uv/pyenv
# But we can check if 'uv' is present as it simplifies things
if ! command -v uv &> /dev/null; then
    echo "ℹ️  'uv' package manager not found. Highly recommended for fast setup."
    # brew install uv output
else
    echo "✅ uv found."
fi

echo "🎉 Dependency check complete!"
echo "If this is a fresh install, remember to run: uv sync (or pip install -e .)"
