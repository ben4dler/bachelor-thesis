import mne

file_path = "./data/raw/ds005363/sub-101/ses-01/eeg/sub-101_ses-01_task-objrecog_channels.tsv"

with open(file_path, "r") as file:
    raw_data = mne.io.read_raw_brainvision(file)


