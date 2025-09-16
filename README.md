---

# Jazz-Based Musical Style Transfer Using Deep Reinforcement Learning

This repository contains the code and findings for the research project, "Jazz-Based Musical Style Transfer," a final year project for the Bachelor of Computer Science at Tunku Abdul Rahman University of Management and Technology (TAR UMT).

The project explores a novel framework for transforming non-jazz piano MIDI files into stylistically authentic jazz pieces by leveraging a combination of deep generative models and reinforcement learning.

---

## Table of Contents
- [The Problem](#the-problem)
- [Solution](#solution)
- [Key Features](#key-features)
- [Methodology Overview](#methodology-overview)
- [Model Architectures](#model-architectures)
- [Results & Key Findings](#results--key-findings)
- [Technology Stack](#technology-stack)
- [Getting Started](#getting-started)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## The Problem

Jazz, a rich and historically significant musical genre, has seen a decline in mainstream popularity, partly due to its harmonic and rhythmic complexity which can be challenging for new listeners and musicians. Current AI-powered music tools primarily focus on generating new compositions from scratch, leaving a significant gap in **musical style transfer**—the ability to reinterpret an existing piece in a new style.

This project addresses three core challenges:
1.  **Cultural Preservation:** Creating tools to make jazz more accessible and encourage new interpretations, helping to revitalize its cultural relevance.
2.  **The Style-Content Trade-off:** Existing models often struggle to infuse a new style (like jazz) without destroying the melodic and harmonic structure of the original piece.
3.  **Lack of Specialized Tools:** There is a need for AI systems designed specifically to learn and transfer the nuanced, improvisational elements of jazz.

## Solution

This project introduces a multi-stage framework to tackle the style transfer challenge. We develop, train, and compare three different deep learning architectures, which are then fine-tuned using **Proximal Policy Optimization (PPO)**, a reinforcement learning algorithm.

The core of the system is a **hybrid reward signal** that teaches the model to balance two competing goals:
- **Stylistic Accuracy (65%):** How "jazz-like" does the output sound? This is measured by a custom-trained jazz genre classifier.
- **Content Preservation (35%):** How well does the output retain the original melody and harmony? This is measured using chroma similarity.

By fine-tuning the models with this reward, they learn to reinterpret a piece with jazz idioms like swing rhythms and extended harmonies while keeping the original song recognizable.

## Key Features

- **Multi-Model Comparison:** Implements and evaluates three distinct generative models:
    1.  Generative Adversarial Network (GAN)
    2.  Relativistic Average GAN (RaGAN)
    3.  Transformer-Variational Autoencoder (Transformer-VAE)
- **Reinforcement Learning for Music:** Utilizes Proximal Policy Optimization (PPO) to dynamically fine-tune the generative models, a novel approach in musical style transfer.
- **Custom Jazz Classifier:** A lightweight CNN trained on a diverse dataset to act as an effective "judge" for the reinforcement learning agent.
- **Efficient and Practical:** The final model can transform a 3-minute MIDI file in under 30 seconds on consumer-grade hardware.

## Methodology Overview

The project workflow is divided into four main stages:

1.  **Data Preparation:** MIDI files from the **ADL, PIAST, and PiJAMA** datasets are preprocessed. This includes outlier removal, transposition to C major, and conversion into 2D piano roll matrices.
2.  **Classifier Training:** A CNN model is trained to accurately classify MIDI segments as "jazz" or "non-jazz." A high-recall model was developed to ensure it could effectively guide the RL process.
3.  **Generative Pre-training:** The GAN, RaGAN, and Transformer-VAE models are pre-trained on a mix of jazz and non-jazz music to develop a foundational understanding of musical structure.
4.  **PPO Fine-Tuning:** The pre-trained generators are treated as policy networks and fine-tuned with PPO, using the hybrid reward signal from the classifier and chroma analysis to learn the style transfer task.

## Model Architectures

- **Jazz Genre Classifier:** A lightweight CNN with asymmetric kernels and dilated convolutions, optimized to capture the pitch-time relationships and rhythmic motifs characteristic of jazz from piano roll matrices.
- **GAN / RaGAN Generator:** A hybrid LSTM-Conv1D architecture that models both sequential dependencies (phrasing) and local harmonic patterns (chords).
- **Transformer-VAE:** A powerful hybrid model that uses a Transformer to capture long-range dependencies in the music, combined with a VAE to learn a structured, disentangled latent space. This allows for more controlled and coherent transformations.

## Results & Key Findings

After a comprehensive evaluation, the **RaGAN-based model demonstrated the most balanced and effective performance.**

- **GAN:** Showed success in adversarial training but was prone to instability and sometimes failed to capture the jazz style consistently.
- **Transformer-VAE:** Was highly effective at adopting the jazz style but tended to be overly aggressive, sometimes sacrificing too much of the original piece's harmonic structure.
- **RaGAN:** Emerged as the most robust solution. It consistently achieved high genre scores while maintaining strong chroma similarity, striking the best balance between stylistic transformation and content preservation. It achieved a **peak average reward of approximately 0.61**

The RaGAN model provides a stable, high-quality solution for transforming non-jazz piano into musically coherent and stylistically authentic jazz.

## Technology Stack

- **Language:** Python 3.13
- **Deep Learning Framework:** PyTorch 2.6
- **Environment Management:** Miniconda
- **Core Libraries:** NumPy, Pandas, Matplotlib
- **MIDI Processing:** `pretty_midi`, `pypianoroll`
- **Evaluation:** Scikit-learn, Google Forms

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details. This allows for free use, modification, and distribution of the code for any purpose.
