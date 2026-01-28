"""
Phoneme-Based Lip Sync System for CoFoc

Converts text to phonemes and maps them to visemes (mouth shapes) for
realistic speech animation. Uses the ARPAbet phoneme set and standard
viseme mappings used in animation and VTubing.

Viseme Categories (based on industry standards):
- sil: Silence/neutral
- PP: P, B, M (bilabial closure)
- FF: F, V (labiodental)
- TH: TH, DH (dental fricative)
- DD: T, D, N, L (alveolar)
- kk: K, G, NG (velar)
- CH: CH, J, SH, ZH (postalveolar)
- SS: S, Z (alveolar fricative)
- nn: N (nasal)
- RR: R (rhotic)
- aa: AA, AH (open vowel)
- E: EH, AE (front mid vowel)
- ih: IH, IY (front high vowel)
- oh: AO, OW (back rounded vowel)
- ou: UH, UW, W (rounded/back vowel)
"""

import re
import time
import threading
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
import math

try:
    from g2p_en import G2p
    G2P_AVAILABLE = True
except ImportError:
    G2P_AVAILABLE = False
    print("Warning: g2p-en not available. Install with: pip install g2p-en")


class Viseme(Enum):
    """Standard viseme categories for lip sync."""
    SIL = "sil"   # Silence - neutral/closed
    PP = "PP"     # P, B, M - lips pressed together
    FF = "FF"     # F, V - lower lip under upper teeth
    TH = "TH"     # TH, DH - tongue between teeth
    DD = "DD"     # T, D, N, L - tongue behind upper teeth
    KK = "kk"     # K, G, NG - back of tongue raised
    CH = "CH"     # CH, J, SH, ZH - lips rounded, tongue up
    SS = "SS"     # S, Z - teeth together, lips slightly open
    NN = "nn"     # N - similar to DD but nasal
    RR = "RR"     # R - lips slightly rounded
    AA = "aa"     # AA, AH - mouth wide open
    E = "E"       # EH, AE - mouth open, spread
    IH = "ih"     # IH, IY, Y - mouth slightly open, spread
    OH = "oh"     # AO, OW - lips rounded, medium open
    OU = "ou"     # UH, UW, W - lips rounded/puckered


@dataclass
class VisemeShape:
    """
    Mouth shape parameters for a viseme.

    Each parameter is 0.0 to 1.0:
    - jaw_open: How far the jaw is dropped
    - lip_pucker: How rounded/puckered the lips are
    - mouth_wide: How wide/stretched the mouth is
    - upper_lip_up: Upper lip raised (showing teeth)
    - lower_lip_down: Lower lip lowered
    - tongue_out: Tongue visibility (for TH)
    """
    jaw_open: float = 0.0
    lip_pucker: float = 0.0
    mouth_wide: float = 0.0
    upper_lip_up: float = 0.0
    lower_lip_down: float = 0.0
    tongue_out: float = 0.0

    def lerp(self, other: 'VisemeShape', t: float) -> 'VisemeShape':
        """Linearly interpolate between two viseme shapes."""
        return VisemeShape(
            jaw_open=self.jaw_open + (other.jaw_open - self.jaw_open) * t,
            lip_pucker=self.lip_pucker + (other.lip_pucker - self.lip_pucker) * t,
            mouth_wide=self.mouth_wide + (other.mouth_wide - self.mouth_wide) * t,
            upper_lip_up=self.upper_lip_up + (other.upper_lip_up - self.upper_lip_up) * t,
            lower_lip_down=self.lower_lip_down + (other.lower_lip_down - self.lower_lip_down) * t,
            tongue_out=self.tongue_out + (other.tongue_out - self.tongue_out) * t,
        )

    def to_simple(self) -> float:
        """Convert to a simple mouth openness value (0-1) for basic avatars."""
        # Weighted combination for overall "openness"
        return min(1.0, self.jaw_open * 0.7 + self.mouth_wide * 0.2 + self.upper_lip_up * 0.1)


# Viseme shape definitions - carefully tuned for realistic mouth movement
VISEME_SHAPES: Dict[Viseme, VisemeShape] = {
    Viseme.SIL: VisemeShape(jaw_open=0.0, lip_pucker=0.0, mouth_wide=0.0),
    Viseme.PP:  VisemeShape(jaw_open=0.0, lip_pucker=0.3, mouth_wide=0.0),  # Lips pressed
    Viseme.FF:  VisemeShape(jaw_open=0.1, lip_pucker=0.0, mouth_wide=0.1, lower_lip_down=0.3),  # F, V
    Viseme.TH:  VisemeShape(jaw_open=0.15, lip_pucker=0.0, mouth_wide=0.2, tongue_out=0.5),  # Tongue out
    Viseme.DD:  VisemeShape(jaw_open=0.2, lip_pucker=0.0, mouth_wide=0.3),  # T, D
    Viseme.KK:  VisemeShape(jaw_open=0.25, lip_pucker=0.0, mouth_wide=0.2),  # K, G
    Viseme.CH:  VisemeShape(jaw_open=0.2, lip_pucker=0.4, mouth_wide=0.0),  # CH, SH - rounded
    Viseme.SS:  VisemeShape(jaw_open=0.1, lip_pucker=0.0, mouth_wide=0.4, upper_lip_up=0.2),  # S, Z - teeth
    Viseme.NN:  VisemeShape(jaw_open=0.15, lip_pucker=0.0, mouth_wide=0.2),  # N
    Viseme.RR:  VisemeShape(jaw_open=0.2, lip_pucker=0.3, mouth_wide=0.1),  # R - slightly rounded
    Viseme.AA:  VisemeShape(jaw_open=0.9, lip_pucker=0.0, mouth_wide=0.5),  # AH - wide open
    Viseme.E:   VisemeShape(jaw_open=0.5, lip_pucker=0.0, mouth_wide=0.7),  # EH - open, spread
    Viseme.IH:  VisemeShape(jaw_open=0.25, lip_pucker=0.0, mouth_wide=0.6),  # IH, IY - smile-ish
    Viseme.OH:  VisemeShape(jaw_open=0.6, lip_pucker=0.5, mouth_wide=0.0),  # OH - rounded open
    Viseme.OU:  VisemeShape(jaw_open=0.3, lip_pucker=0.8, mouth_wide=0.0),  # OO - puckered
}


# ARPAbet phoneme to viseme mapping
PHONEME_TO_VISEME: Dict[str, Viseme] = {
    # Silence
    '': Viseme.SIL,
    ' ': Viseme.SIL,
    '.': Viseme.SIL,
    ',': Viseme.SIL,
    '?': Viseme.SIL,
    '!': Viseme.SIL,

    # Bilabial (PP) - lips together
    'P': Viseme.PP,
    'B': Viseme.PP,
    'M': Viseme.PP,

    # Labiodental (FF) - teeth on lip
    'F': Viseme.FF,
    'V': Viseme.FF,

    # Dental (TH) - tongue between teeth
    'TH': Viseme.TH,
    'DH': Viseme.TH,

    # Alveolar (DD) - tongue behind teeth
    'T': Viseme.DD,
    'D': Viseme.DD,
    'N': Viseme.DD,  # Could also be NN
    'L': Viseme.DD,

    # Velar (KK) - back of tongue
    'K': Viseme.KK,
    'G': Viseme.KK,
    'NG': Viseme.KK,

    # Postalveolar (CH) - lips rounded
    'CH': Viseme.CH,
    'JH': Viseme.CH,
    'SH': Viseme.CH,
    'ZH': Viseme.CH,

    # Alveolar fricative (SS)
    'S': Viseme.SS,
    'Z': Viseme.SS,

    # Rhotic (RR)
    'R': Viseme.RR,
    'ER': Viseme.RR,
    'ER0': Viseme.RR,
    'ER1': Viseme.RR,
    'ER2': Viseme.RR,

    # Approximants
    'W': Viseme.OU,
    'Y': Viseme.IH,
    'HH': Viseme.SIL,  # H is almost invisible

    # Vowels - Open (AA)
    'AA': Viseme.AA,
    'AA0': Viseme.AA,
    'AA1': Viseme.AA,
    'AA2': Viseme.AA,
    'AH': Viseme.AA,
    'AH0': Viseme.AA,
    'AH1': Viseme.AA,
    'AH2': Viseme.AA,

    # Vowels - Front mid (E)
    'EH': Viseme.E,
    'EH0': Viseme.E,
    'EH1': Viseme.E,
    'EH2': Viseme.E,
    'AE': Viseme.E,
    'AE0': Viseme.E,
    'AE1': Viseme.E,
    'AE2': Viseme.E,

    # Vowels - Front high (IH)
    'IH': Viseme.IH,
    'IH0': Viseme.IH,
    'IH1': Viseme.IH,
    'IH2': Viseme.IH,
    'IY': Viseme.IH,
    'IY0': Viseme.IH,
    'IY1': Viseme.IH,
    'IY2': Viseme.IH,

    # Vowels - Back rounded (OH)
    'AO': Viseme.OH,
    'AO0': Viseme.OH,
    'AO1': Viseme.OH,
    'AO2': Viseme.OH,
    'OW': Viseme.OH,
    'OW0': Viseme.OH,
    'OW1': Viseme.OH,
    'OW2': Viseme.OH,
    'OY': Viseme.OH,
    'OY0': Viseme.OH,
    'OY1': Viseme.OH,
    'OY2': Viseme.OH,

    # Vowels - Rounded/back (OU)
    'UH': Viseme.OU,
    'UH0': Viseme.OU,
    'UH1': Viseme.OU,
    'UH2': Viseme.OU,
    'UW': Viseme.OU,
    'UW0': Viseme.OU,
    'UW1': Viseme.OU,
    'UW2': Viseme.OU,

    # Diphthongs
    'AW': Viseme.AA,  # Start open, moves to OU
    'AW0': Viseme.AA,
    'AW1': Viseme.AA,
    'AW2': Viseme.AA,
    'AY': Viseme.AA,  # Start open, moves to IH
    'AY0': Viseme.AA,
    'AY1': Viseme.AA,
    'AY2': Viseme.AA,
    'EY': Viseme.E,   # Start E, moves to IH
    'EY0': Viseme.E,
    'EY1': Viseme.E,
    'EY2': Viseme.E,
}


@dataclass
class PhonemeEvent:
    """A single phoneme with timing information."""
    phoneme: str
    viseme: Viseme
    start_time: float  # Seconds from start
    duration: float    # Seconds

    @property
    def end_time(self) -> float:
        return self.start_time + self.duration


class LipSyncEngine:
    """
    Converts text to timed phoneme sequences for lip sync animation.
    """

    # Average phoneme durations in seconds (can be adjusted)
    CONSONANT_DURATION = 0.08
    VOWEL_DURATION = 0.12
    SILENCE_DURATION = 0.15

    def __init__(self):
        self.g2p = None
        if G2P_AVAILABLE:
            try:
                self.g2p = G2p()
                print("Lip sync: G2P engine initialized")
            except Exception as e:
                print(f"Warning: Could not initialize G2P: {e}")

        # Current playback state
        self._events: List[PhonemeEvent] = []
        self._start_time: float = 0.0
        self._is_playing: bool = False
        self._lock = threading.Lock()

    def text_to_phonemes(self, text: str) -> List[str]:
        """Convert text to ARPAbet phonemes."""
        if self.g2p is None:
            # Fallback: simple vowel detection
            return self._simple_phoneme_fallback(text)

        try:
            # G2P returns a list of phonemes
            phonemes = self.g2p(text)
            # Filter and clean
            result = []
            for p in phonemes:
                p = p.strip()
                if p and p not in [' ', '', '-']:
                    # Remove stress markers for lookup but keep for reference
                    result.append(p.upper())
            return result
        except Exception as e:
            print(f"G2P error: {e}")
            return self._simple_phoneme_fallback(text)

    def _simple_phoneme_fallback(self, text: str) -> List[str]:
        """Simple fallback when G2P is not available."""
        phonemes = []
        vowels = set('aeiouAEIOU')
        text = text.lower()

        i = 0
        while i < len(text):
            c = text[i]

            if c in ' .,!?;:':
                phonemes.append(' ')
            elif c in vowels:
                # Map vowels to approximate phonemes
                vowel_map = {'a': 'AA', 'e': 'EH', 'i': 'IY', 'o': 'OW', 'u': 'UW'}
                phonemes.append(vowel_map.get(c, 'AH'))
            elif c == 't' and i + 1 < len(text) and text[i + 1] == 'h':
                phonemes.append('TH')
                i += 1
            elif c == 's' and i + 1 < len(text) and text[i + 1] == 'h':
                phonemes.append('SH')
                i += 1
            elif c == 'c' and i + 1 < len(text) and text[i + 1] == 'h':
                phonemes.append('CH')
                i += 1
            elif c.isalpha():
                # Map consonants
                consonant_map = {
                    'b': 'B', 'c': 'K', 'd': 'D', 'f': 'F', 'g': 'G',
                    'h': 'HH', 'j': 'JH', 'k': 'K', 'l': 'L', 'm': 'M',
                    'n': 'N', 'p': 'P', 'q': 'K', 'r': 'R', 's': 'S',
                    't': 'T', 'v': 'V', 'w': 'W', 'x': 'K', 'y': 'Y', 'z': 'Z'
                }
                phonemes.append(consonant_map.get(c, 'AH'))

            i += 1

        return phonemes

    def phoneme_to_viseme(self, phoneme: str) -> Viseme:
        """Convert a phoneme to its corresponding viseme."""
        # Strip stress markers (numbers at end)
        base_phoneme = re.sub(r'\d+$', '', phoneme.upper())
        return PHONEME_TO_VISEME.get(phoneme.upper(),
               PHONEME_TO_VISEME.get(base_phoneme, Viseme.SIL))

    def create_phoneme_timeline(self, text: str, total_duration: float) -> List[PhonemeEvent]:
        """
        Create a timed sequence of phoneme events for the given text.

        Args:
            text: The text being spoken
            total_duration: Total duration of the audio in seconds

        Returns:
            List of PhonemeEvent with timing information
        """
        phonemes = self.text_to_phonemes(text)

        if not phonemes:
            return [PhonemeEvent(' ', Viseme.SIL, 0.0, total_duration)]

        # Calculate base duration per phoneme
        # Vowels are typically longer than consonants
        vowel_phonemes = {'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY',
                         'IH', 'IY', 'OW', 'OY', 'UH', 'UW'}

        total_weight = 0
        for p in phonemes:
            base = re.sub(r'\d+$', '', p.upper())
            if base in vowel_phonemes:
                total_weight += 1.5  # Vowels are longer
            elif p.strip() == '' or p in ' .,!?':
                total_weight += 0.5  # Pauses are shorter
            else:
                total_weight += 1.0

        # Distribute time proportionally
        time_per_weight = total_duration / max(total_weight, 1)

        events = []
        current_time = 0.0

        for p in phonemes:
            base = re.sub(r'\d+$', '', p.upper())

            if base in vowel_phonemes:
                duration = time_per_weight * 1.5
            elif p.strip() == '' or p in ' .,!?':
                duration = time_per_weight * 0.5
            else:
                duration = time_per_weight * 1.0

            viseme = self.phoneme_to_viseme(p)
            events.append(PhonemeEvent(
                phoneme=p,
                viseme=viseme,
                start_time=current_time,
                duration=duration
            ))
            current_time += duration

        return events

    def start_playback(self, text: str, duration: float) -> None:
        """Start lip sync playback for the given text and duration."""
        with self._lock:
            self._events = self.create_phoneme_timeline(text, duration)
            self._start_time = time.time()
            self._is_playing = True

    def stop_playback(self) -> None:
        """Stop lip sync playback."""
        with self._lock:
            self._is_playing = False
            self._events = []

    def get_current_viseme(self) -> Tuple[VisemeShape, float]:
        """
        Get the current viseme shape based on playback time.

        Returns:
            Tuple of (VisemeShape, blend_amount) where blend_amount
            indicates transition progress (0-1).
        """
        with self._lock:
            if not self._is_playing or not self._events:
                return VISEME_SHAPES[Viseme.SIL], 0.0

            elapsed = time.time() - self._start_time

            # Find current and next phoneme
            current_event = None
            next_event = None

            for i, event in enumerate(self._events):
                if event.start_time <= elapsed < event.end_time:
                    current_event = event
                    if i + 1 < len(self._events):
                        next_event = self._events[i + 1]
                    break

            if current_event is None:
                # Past the end or before start
                if elapsed >= self._events[-1].end_time:
                    self._is_playing = False
                return VISEME_SHAPES[Viseme.SIL], 0.0

            # Get current shape
            current_shape = VISEME_SHAPES[current_event.viseme]

            # Calculate blend to next shape for smooth transition
            if next_event:
                # Blend in the last 30% of the current phoneme
                blend_start = current_event.start_time + current_event.duration * 0.7
                if elapsed > blend_start:
                    blend_t = (elapsed - blend_start) / (current_event.duration * 0.3)
                    blend_t = min(1.0, max(0.0, blend_t))
                    # Smooth step for more natural transition
                    blend_t = blend_t * blend_t * (3 - 2 * blend_t)
                    next_shape = VISEME_SHAPES[next_event.viseme]
                    current_shape = current_shape.lerp(next_shape, blend_t)

            return current_shape, 1.0

    def get_simple_mouth_open(self) -> float:
        """Get a simple 0-1 mouth openness value for basic avatars."""
        shape, _ = self.get_current_viseme()
        return shape.to_simple()

    @property
    def is_playing(self) -> bool:
        """Check if lip sync is currently playing."""
        with self._lock:
            return self._is_playing


# Singleton instance for easy access
_lip_sync_engine: Optional[LipSyncEngine] = None


def get_lip_sync_engine() -> LipSyncEngine:
    """Get or create the global lip sync engine."""
    global _lip_sync_engine
    if _lip_sync_engine is None:
        _lip_sync_engine = LipSyncEngine()
    return _lip_sync_engine
