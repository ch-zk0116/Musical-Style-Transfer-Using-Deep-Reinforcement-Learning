# PenaltyCalculation.py
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# ==============================================================================
# SECTION 1: DEFAULT CONFIGURATION
# ==============================================================================
class DefaultPenaltyConfig:
    """
    A standalone class holding default parameters for musicality calculations.
    This is used when MusicalityAndChromaLosses is instantiated without a config object.
    """
    # Device and Shape Parameters
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    PITCHES = 88
    TIMESTEPS = 128
    
    # Musical Rule Parameters
    WINDOW_SIZE = 64
    LOSS_THRESHOLD_STEEPNESS = 15.0
    SILENT_VELOCITY_THRESHOLD = 40
    MIN_NOTES_IN_WINDOW = 5

    # Loss Weights (provide sensible defaults)
    W_PENALTY_IMPOSSIBLE = 0.5
    W_PENALTY_SILENT_ONSET = 1.0
    W_PENALTY_ORPHAN_VELOCITY = 0.5
    W_MIN_NOTE_RATE = 1.0


# ==============================================================================
# SECTION 2: THE MAIN CLASS
# ==============================================================================
class MusicalityAndChromaLosses:
    def __init__(self, config=None):
        if config is None:
            print("WARNING: No config provided to MusicalityAndChromaLosses. Using internal defaults.")
            self.config = DefaultPenaltyConfig()
        else:
            self.config = config

        self.device = self.config.DEVICE
        self.pitches = self.config.PITCHES
        self.timesteps = self.config.TIMESTEPS
        self.window_size = self.config.WINDOW_SIZE
        self.k = self.config.LOSS_THRESHOLD_STEEPNESS
        
        self.pitch_to_chroma_map = self._create_pitch_to_chroma_map().to(self.device)

    def _create_pitch_to_chroma_map(self):
        midi_pitches = torch.arange(21, 21 + self.pitches)
        chroma_indices = midi_pitches % 12
        pitch_to_chroma_map = torch.zeros(12, self.pitches)
        for i in range(self.pitches):
            pitch_to_chroma_map[chroma_indices[i], i] = 1.0
        return pitch_to_chroma_map
    
    def _apply_soft_threshold(self, probabilities):
        return torch.sigmoid(self.k * (probabilities - 0.5))

    def calculate_impossible_note_penalty(self, generated_batch):
        onset_probs = generated_batch[:, 0, :, :]
        sustain_probs = generated_batch[:, 1, :, :]
        no_sustain_probs = 1.0 - sustain_probs
        impossible_event_probs = onset_probs * no_sustain_probs
        # [MODIFIED] Return a per-sample penalty
        return torch.mean(impossible_event_probs, dim=(1, 2))

    def calculate_silent_onset_penalty(self, generated_batch):
        onset_probs = generated_batch[:, 0, :, :]
        velocity_probs = generated_batch[:, 2, :, :]
        v_thresh_norm = self.config.SILENT_VELOCITY_THRESHOLD / 127.0
        prob_is_silent = 1.0 - torch.sigmoid(self.k * (velocity_probs - v_thresh_norm))
        silent_onset_event_probs = onset_probs * prob_is_silent
        # [MODIFIED] Return a per-sample penalty
        return torch.mean(silent_onset_event_probs, dim=(1, 2))

    def calculate_orphan_velocity_penalty(self, generated_batch):
        onset_probs = generated_batch[:, 0, :, :]
        velocity_probs = generated_batch[:, 2, :, :]
        no_onset_probs = 1.0 - onset_probs
        orphan_event_probs = velocity_probs * no_onset_probs
        # [MODIFIED] Return a per-sample penalty
        return torch.mean(orphan_event_probs, dim=(1, 2))

    def _get_confident_notes_in_window(self, generated_batch):
        onset_probs = generated_batch[:, 0:1, :, :]
        soft_onsets = self._apply_soft_threshold(onset_probs)
        onsets_per_timestep = torch.sum(soft_onsets, dim=2)
        kernel = torch.ones(1, 1, self.window_size, device=self.device)
        return F.conv1d(onsets_per_timestep, kernel, padding='same')

    def calculate_min_notes_penalty(self, generated_batch):
        confident_notes_in_window = self._get_confident_notes_in_window(generated_batch)
        violations = F.relu(self.config.MIN_NOTES_IN_WINDOW - confident_notes_in_window)
        # [MODIFIED] Return a per-sample penalty (average over the time dimension)
        return torch.mean(violations, dim=2).squeeze(1)

    def get_musicality_losses(self, generated_batch):
        losses = {
            'impossible': self.calculate_impossible_note_penalty(generated_batch),
            'silent_onset': self.calculate_silent_onset_penalty(generated_batch),
            'orphan_velocity': self.calculate_orphan_velocity_penalty(generated_batch),
            'min_rate': self.calculate_min_notes_penalty(generated_batch),
        }

        # [MODIFIED] This now correctly performs element-wise multiplication and summation,
        # resulting in a per-sample total weighted loss tensor.
        total_weighted_loss = (
            self.config.W_PENALTY_IMPOSSIBLE * losses['impossible'] +
            self.config.W_PENALTY_SILENT_ONSET * losses['silent_onset'] +
            self.config.W_PENALTY_ORPHAN_VELOCITY * losses['orphan_velocity'] +
            self.config.W_MIN_NOTE_RATE * losses['min_rate']  
        )
        return losses, total_weighted_loss

# ==============================================================================
# SECTION 3: STANDALONE UTILITY FUNCTIONS
# ==============================================================================
def plot_piano_roll(matrices, threshold):
    onset_matrix = matrices["onset"]
    sustain_matrix = matrices["sustain"]
    full_note_matrix = np.logical_or(onset_matrix, sustain_matrix).astype(int)
    plt.figure(figsize=(16, 8))
    plt.imshow(full_note_matrix, aspect='auto', origin='lower', cmap='viridis')
    plt.title(f'Piano Roll for Threshold: {threshold}')
    plt.xlabel('Time Steps')
    plt.ylabel('Pitch (MIDI Note - 21)')
    plt.colorbar(label='Note Active')
    plt.grid(False)
    plt.show()