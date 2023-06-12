import os

# Define the musical styles
genre = [
    'action',
    'adventure',
    'arcade',
    'horror'
]

styles = [
    [
        'data/action/batman',
        'data/action/kung_fu'
    ],
    [
        'data/adventure/blade_runner',
        'data/adventure/myst'
    ],
    [
        'data/arcade/blox',
        'data/arcade/burning_monkey',
        'data/arcade/mario'
    ],
    [
        'data/horror/blood',
        'data/horror/house_of_the_dead'
    ]
]

NUM_STYLES = sum(len(s) for s in styles)

NUM_INSTRUMENTS = 10
MAX_INSTRUMENTS_PER_SONG = 1
MAX_INSTRUMENTS_GENERATED = 10
FS = 16

# MIDI Resolution
DEFAULT_RES = 96
MIDI_MAX_NOTES = 128
MAX_VELOCITY = 127

# Number of octaves supported
NUM_OCTAVES = 4
OCTAVE = 8

# Min and max note (in MIDI note number)
MIN_NOTE = 30
# MAX_NOTE = MIN_NOTE + NUM_OCTAVES * OCTAVE
MAX_NOTE = 85
NUM_NOTES_INSTRUMENT = MAX_NOTE - MIN_NOTE + 1
NUM_NOTES = (NUM_INSTRUMENTS + 1) * NUM_NOTES_INSTRUMENT

# Number of beats in a bar
BEATS_PER_BAR = 4
# Notes per quarter note
NOTES_PER_BEAT = 4
# The quickest note is a half-note
# NOTES_PER_BAR = NOTES_PER_BEAT * BEATS_PER_BAR
NOTE_TIME_STEPS = 2
NOTES_PER_BAR = NOTES_PER_BEAT * BEATS_PER_BAR * NOTE_TIME_STEPS

# Training parameters
BARS = 1
BATCH_SIZE = 32
SEQ_LEN = int(BARS * NOTES_PER_BAR)
ALLOW_SAVE = False

# Hyper Parameters
ENCODER_UNITS = 512
ENCODER_UNITS_2 = 256
ENCODER_UNITS_3 = 128
LATENT_DIM = 64
BETA = 0.01
EPOCHS = 3001
GENERATE_EVERY_EPOCH = 50000

# Move file save location
OUT_DIR = 'out'
MODEL_DIR = os.path.join(OUT_DIR, 'models')
MODEL_FILE = os.path.join(OUT_DIR, 'model.h5')
SAMPLES_DIR = os.path.join(OUT_DIR, 'samples')
CACHE_DIR = os.path.join(OUT_DIR, 'cache')
