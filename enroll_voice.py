"""Enroll a single owner voice profile for local biometric access control.

Usage:
    python enroll_voice.py
    python enroll_voice.py --profile user_voice_profile.npy --count 5
"""

from __future__ import annotations

import argparse
from pathlib import Path

from voice_auth import VoiceAuthenticator


def main() -> None:
    parser = argparse.ArgumentParser(description="Record a few spoken phrases and save the owner's voice embedding.")
    parser.add_argument("--profile", default="user_voice_profile.npy", help="Path to the saved voice profile (.npy).")
    parser.add_argument("--count", type=int, default=4, help="Number of utterances to record for enrollment.")
    parser.add_argument("--threshold", type=float, default=0.75, help="Cosine-similarity threshold used for verification.")
    parser.add_argument("--device-index", type=int, default=-1, help="Optional microphone device index to use.")
    args = parser.parse_args()

    authenticator = VoiceAuthenticator(
        profile_path=Path(args.profile),
        threshold=args.threshold,
        device_index=args.device_index,
    )
    embedding = authenticator.enroll_user(record_count=max(3, min(args.count, 5)), phrase_prompt="Jarvis, register my voice")
    print(f"Enrollment complete. Voice profile saved to {Path(args.profile)}")
    print(f"Embedding dimension: {embedding.shape[0]}")


if __name__ == "__main__":
    main()
