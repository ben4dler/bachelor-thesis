import mne


def get_events(raw):
    # Returns the events_arr with form [timepoint, previous value, event-code]
    # And return the event_dict with form Dict[String, int] with String being the stimulus description and int being the event number
    events_arr, event_dict = mne.events_from_annotations(raw)
    return events_arr, event_dict


def get_categories(number_of_categories=None):

    # Creates a dict[string, range(int)] corresponding to the categories used in this dataset.
    # Param number_of_categories list[string] Defaults to none, means all categories in this dataset are returned with the distribution below.
    # If a list is passed, the categories will be entered into the result in the same order they are passed, with the same distribution as the default value.
    # The Categories are as follows:
    # 1 - 16: Face;
    # 17 - 32: Animal;
    # 33 - 48: Place;
    # 49 - 64: Object;

    all_categories = {
        "face": range(1, 17),
        "animal": range(17, 33),
        "place": range(33, 49),
        "object": range(49, 65),
    }

    if number_of_categories is None:
        return all_categories

    selected_categories = {}

    for name, event_range in all_categories.items():
        if len(selected_categories) == number_of_categories:
            break
        else:
            selected_categories[name] = event_range

    return selected_categories


def create_event_ids(event_dict, categories):
    # Baut das Dictionary, das mne.Epochs als event_id bekommt.
    # Beispiel für einen Eintrag: "face/S12" -> 12
    # Durch den "/" im Namen kann man später mit epochs["face"]
    # alle Epochs einer Kategorie auswählen.
    event_ids = {}

    for category_name, stimulus_numbers in categories.items():
        for stimulus_number in stimulus_numbers:
            marker_name = f"Stimulus/S{stimulus_number:>3}"
            event_code = event_dict[marker_name]

            new_name = f"{category_name}/S{stimulus_number}"
            event_ids[new_name] = event_code

    return event_ids


def create_epochs(raw, events_arr, event_id):
    epochs = mne.Epochs(
        raw,
        events_arr,
        event_id=event_id,
        tmin=-0.2,
        tmax=0.8,
        baseline=((-0.2, 0)),
        preload=True,
        event_repeated="drop",
    )
    return epochs
