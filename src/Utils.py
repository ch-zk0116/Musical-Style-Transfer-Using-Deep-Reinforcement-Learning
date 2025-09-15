# Utils.py
import os
import numpy as np
import pretty_midi
import matplotlib.pyplot as plt


def plot_original_content_matrices(content_tensor, title="Original Content Matrices"):
    """
    Visualizes the three input channels of a single content item.
    """
    print(f"--- Visualizing: {title} ---")
    if content_tensor.is_cuda:
        content_tensor = content_tensor.cpu()
        
    # Convert to numpy and get individual channels
    content_np = content_tensor.numpy()
    onset_matrix = content_np[0].T
    sustain_matrix = content_np[1].T
    velocity_matrix = content_np[2].T
    
    fig, axs = plt.subplots(1, 3, figsize=(24, 7))
    fig.suptitle(title, fontsize=18)

    # --- Plot Onset ---
    im1 = axs[0].imshow(onset_matrix, aspect='auto', origin='lower', cmap='plasma', vmin=0, vmax=1)
    axs[0].set_title('Channel 0: Original Onsets')
    axs[0].set_ylabel('Pitch')
    axs[0].set_xlabel('Time Steps')
    fig.colorbar(im1, ax=axs[0])

    # --- Plot Sustain ---
    im2 = axs[1].imshow(sustain_matrix, aspect='auto', origin='lower', cmap='plasma', vmin=0, vmax=1)
    axs[1].set_title('Channel 1: Original Sustains')
    axs[1].set_xlabel('Time Steps')
    fig.colorbar(im2, ax=axs[1])

    # --- Plot Velocity ---
    im3 = axs[2].imshow(velocity_matrix, aspect='auto', origin='lower', cmap='plasma', vmin=0, vmax=1)
    axs[2].set_title('Channel 2: Original Velocities')
    axs[2].set_xlabel('Time Steps')
    fig.colorbar(im3, ax=axs[2])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    plt.close()

def plot_generator_output_matrices(onset_probs, sustain_probs, velocity_values, title="Generator Raw Output Probabilities"):
    """
    Visualizes the three output channels of the generator as heatmaps.
    This function is self-contained and does not require a config object.
    """
    print(f"--- Visualizing: {title} ---")

    fig, axs = plt.subplots(1, 3, figsize=(24, 7)) # Changed to 1x3 layout
    fig.suptitle(title, fontsize=18)

    # --- Plot Onset Probabilities ---
    im1 = axs[0].imshow(onset_probs.T, aspect='auto', origin='lower', cmap='viridis', vmin=0, vmax=1)
    axs[0].set_title('Channel 0: Onset Probabilities')
    axs[0].set_ylabel('Pitch')
    axs[0].set_xlabel('Time Steps')
    fig.colorbar(im1, ax=axs[0])

    # --- Plot Sustain Probabilities ---
    im2 = axs[1].imshow(sustain_probs.T, aspect='auto', origin='lower', cmap='viridis', vmin=0, vmax=1)
    axs[1].set_title('Channel 1: Sustain Probabilities')
    axs[1].set_xlabel('Time Steps')
    fig.colorbar(im2, ax=axs[1])

    # --- Plot Velocity Values ---
    im3 = axs[2].imshow(velocity_values.T, aspect='auto', origin='lower', cmap='viridis', vmin=0, vmax=1)
    axs[2].set_title('Channel 2: Velocity Values')
    axs[2].set_xlabel('Time Steps')
    fig.colorbar(im3, ax=axs[2])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def reconstruct_and_save_midi(generated_tensor, path, config=None, bpm=120.0):
    """
    MODIFIED: Converts a 3-channel tensor into a MIDI file.
    - Now infers TIMESTEPS and PITCHES directly from the tensor shape.
    - Provides a default value for NOTE_ON_THRESHOLDS if not found in config.
    """
    if generated_tensor.shape[1] != 3: # Check for 3 channels
        print(f"Warning: Invalid tensor shape provided for MIDI reconstruction. Expected 3 channels, got {generated_tensor.shape[1]}. Skipping.")
        return None

    sample_tensor = generated_tensor[0]
    onsets_sustains_velocity = sample_tensor[0:3]
    
    onset_probs_np = onsets_sustains_velocity[0].T
    sustain_probs_np = onsets_sustains_velocity[1].T
    velocity_values_np = onsets_sustains_velocity[2].T
    
    # --- MODIFICATION START: Make function more robust ---
    # 1. Dynamically get dimensions from the data itself, not config.
    TIMESTEPS, PITCHES = onset_probs_np.shape
    
    # 2. Safely get thresholds with a fallback default value.
    # This uses [0.5] if config is None or if the attribute doesn't exist.
    note_on_thresholds = getattr(config, 'NOTE_ON_THRESHOLDS', [0.5])
    # --- MODIFICATION END ---

    plot_generator_output_matrices(onset_probs_np, sustain_probs_np, velocity_values_np, title=f"Generator Raw Output for Sample: {os.path.basename(path)}")

    all_final_matrices = {}

    for threshold in note_on_thresholds:
        reconstructed_onset = np.zeros_like(onset_probs_np)
        reconstructed_sustain = np.zeros_like(sustain_probs_np)
        reconstructed_velocity = np.zeros_like(velocity_values_np)

        pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
        instrument = pretty_midi.Instrument(program=0, name='Acoustic Grand Piano')
        RESOLUTION = 24
        PITCHES_LOW = 21
        tick_duration = 60.0 / (bpm * RESOLUTION)
        min_duration = tick_duration / 2
        active_notes = {}

        for t_step in range(TIMESTEPS):
            for pitch_idx in range(PITCHES):
                if pitch_idx not in active_notes and onset_probs_np[t_step, pitch_idx] > threshold:
                    start_time = t_step * tick_duration
                    velocity_val = velocity_values_np[t_step, pitch_idx]
                    velocity = int(velocity_val * 126) + 1
                    active_notes[pitch_idx] = (start_time, velocity, t_step)
                    reconstructed_onset[t_step, pitch_idx] = 1
                    reconstructed_velocity[t_step, pitch_idx] = velocity_val

                elif pitch_idx in active_notes and sustain_probs_np[t_step, pitch_idx] < threshold:
                    start_time, velocity, start_t_step = active_notes.pop(pitch_idx)
                    end_time = t_step * tick_duration
                    if end_time - start_time >= min_duration:
                        instrument.notes.append(pretty_midi.Note(
                            velocity=max(1, min(127, velocity)),
                            pitch=pitch_idx + PITCHES_LOW,
                            start=start_time,
                            end=end_time
                        ))
                    reconstructed_sustain[start_t_step:t_step, pitch_idx] = 1

        for pitch_idx, (start_time, velocity, start_t_step) in active_notes.items():
            end_time = TIMESTEPS * tick_duration
            if end_time - start_time >= min_duration:
                instrument.notes.append(pretty_midi.Note(
                    velocity=max(1, min(127, velocity)),
                    pitch=pitch_idx + PITCHES_LOW,
                    start=start_time,
                    end=end_time
                ))
            reconstructed_sustain[start_t_step:TIMESTEPS, pitch_idx] = 1

        pm.instruments.append(instrument)
        file_path = f"{path}_thresh_{threshold:.2f}.mid"
        pm.write(file_path)

        final_matrices = {
            "onset": reconstructed_onset.T,
            "sustain": reconstructed_sustain.T,
            "velocity": reconstructed_velocity.T,
        }
        all_final_matrices[threshold] = final_matrices

    return all_final_matrices