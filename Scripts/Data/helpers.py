def resample_and_rename(df, srate: str, name):
    """
    Simple resampling function that renames dataframes
    :param df: Pandas dataframe with data to be resampled
    :param srate: Samplerate with s at the end
    :param name: Name of current dataset e.g., density
    """
    # Special case where srate is equal to base
    if srate == "base":
        print("Base sample_rate retained")
        return (df, f"{name}_base_cadence")

    # Calculate current cadence in seconds
    cadence = int((df.index[1] - df.index[0]).total_seconds())
    assert srate[-1] == "s", "Please state objective sample rate in seconds e.g., '60s'"
    obj_cad = int(srate[:-1])

    # When cadence too slow
    if cadence > obj_cad:
        print("Filling in missing values")
        df = df.resample(srate).interpolate()
        flag = f"upsampled_from_{cadence}s"

    # When cadence too fast
    elif cadence < obj_cad:
        print("Averaging values")
        df = df.resample(srate).mean()
        flag = f"downsampled_from_{cadence}s"

    # If cadence equal
    elif cadence == obj_cad:
        print(f"Nothing done, cadence matches {srate}")
        flag = "unchanged"

    else:
        raise ValueError(f"Cadence of df : {cadence} Cannot be resampled to {obj_cad}")

    return (df, f"{name}_{flag}")
