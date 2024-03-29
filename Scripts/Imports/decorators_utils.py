def trace(func):
    """
    Print out trace information from a function: Inputs and outputs
    """

    def wrapper(*args, **kwargs):
        print(f"\n Trace: calling {func.__name__}() " f"with {args}, {kwargs} \n")
        original_result = func(*args, **kwargs)

        print(f"Trace: {func.__name__}() " f"returned {original_result!r}")

        return original_result

    return wrapper
