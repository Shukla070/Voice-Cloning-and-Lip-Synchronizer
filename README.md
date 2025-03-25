# Voice Cloning and Lip Synchronizer

## Mentors
M Uzzwal
Rahul Bhimkari

## Mentees
Aadharsh Ramachandran
Gauri Aggarwal
Parihasa K Reddy
Utkarsh Shukla

## Aim
The project aims to develop a system that generates realistic speech through voice cloning and synchronizes lip movements with minimal input data, enabling seamless dubbing, virtual avatars, and content creation.

## Introduction and Overview
Voice cloning and lip synchronization have significant applications in AI-driven content creation, dubbing, and virtual avatars. This project utilizes Tortoise TTS for zero-shot voice cloning, requiring just a few short audio clips (5-10 seconds) from a speaker to generate realistic speech. The generated voice is then processed by Wav2Lip to produce a synchronized video where the speaker’s lip movements match the cloned voice, ensuring natural and high-quality speech animation.

## Technologies Used
Voice Cloning: Tortoise TTS (zero-shot voice cloning)

Lip Syncing: Wav2Lip (modifies lip movements to match generated speech)

Frameworks & Libraries: PyTorch, OpenCV, FFmpeg

Backend: Flask for serving the application

Evaluation Metrics: MOS (Mean Opinion Score), SyncNet Scores

## Datasets
Short audio clips (5-10 sec) of different speakers for training/testing.

Video samples used to evaluate lip-sync accuracy with cloned speech.

Open-source datasets for TTS and lip-syncing models (such as LRS2, VoxCeleb).

## Model and Architecture
Tortoise TTS for generating high-quality, expressive speech from limited data.

Wav2Lip for modifying lip movements to synchronize with the cloned voice.

Modular Deep Learning Pipeline:

Input audio processing

TTS-based speech generation

Lip-syncing using Wav2Lip

Video output generation

## Gradio Interface
Input text for speech synthesis.

Choose your favourite speaker.

Preview the final synchronized video.

## Conclusion
This project successfully integrates zero-shot voice cloning with lip synchronization, producing natural and expressive speech-driven animations. It enhances AI-based dubbing, virtual avatars, and personalized content creation, offering an innovative approach to multimedia generation.

