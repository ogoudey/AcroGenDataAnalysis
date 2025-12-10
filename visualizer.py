import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_values_from_data(csv_path, header="average_pupil_diameter_mm"):
    """
    Returns `header` data column from the CSV.
    """
	
    df = pd.read_csv(csv_path)
    return df[header].values

def fit_increments(inc):
		"""Returns a pupil size estimator given the sampled pupil sizes."""
		x = np.linspace(0, 1, len(inc), endpoint=True)
		y = inc
		coeffs = np.polyfit(x, y, 3)
		return np.poly1d(coeffs)

def clean_pupil_sizes(pupil_sizes):
    """Interpolates missing data"""
    arr = np.array(pupil_sizes, dtype=float)
    arr[(arr < 0.8) | (arr > 10)] = np.nan
    valid = ~np.isnan(arr)
    x = np.arange(len(arr))
    cleaned = np.interp(x, x[valid], arr[valid])

    return cleaned

def calculate(pupil_sizes, brightnesses, predict_function):
    """
    Calculate the "average power".

    Does `pupil_sizes` - `predicted_pupil_sizes`

    Args:
        pupil_sizes: The actual recorded pupil sizes
        brightnesses: The recorded brightness of the screen (previously called "luminance")
        predict_function: The "calibrated" pupil_size estimator

    """
    cleaned_pupil_sizes = clean_pupil_sizes(pupil_sizes)
    input(f"{cleaned_pupil_sizes} (Hold [enter])")
    input(brightnesses)
    predicted_pupil_sizes = np.array( [predict_function(l) for l in brightnesses] )
    input(predicted_pupil_sizes)
    difference = cleaned_pupil_sizes - predicted_pupil_sizes
    input(difference)
    N = 2
    y_padded = np.pad(difference, (N//2, N-1-N//2), mode='edge')
    input(y_padded)
    predicted_pupil_size_moving_avg = np.convolve(y_padded, np.ones((N,))/N, mode='valid')
    input(predicted_pupil_size_moving_avg)
    power = predicted_pupil_size_moving_avg ** 2
    input(power)
    window_size = 2
    moving_avg_power = np.convolve(power, np.ones(window_size)/window_size, mode='valid')
    input(moving_avg_power)
    return difference, moving_avg_power




def main(csv_path="data/Calibration_log_11251606_001.csv"):
    # Splits are within data (assumed)

    # --- Calibrate ---
    increments = get_values_from_data(csv_path)[:17]
    predict_function = fit_increments(increments)
    brightnesses = get_values_from_data(csv_path, "screen_brightness")[:17]

    # --- Plot calibration ---
    plt.figure()
    plt.plot(increments, label="Pupil Size (Increments)", linewidth=2)
    plt.plot(brightnesses, label="Brightness", linewidth=2)
    plt.plot(np.array( [predict_function(l) for l in brightnesses] ), label="Prediction function", linewidth=2)
    plt.title("Calibration")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- Calculate ---
    pupil_sizes = get_values_from_data(csv_path)[18:]
    brightnesses = get_values_from_data(csv_path, "screen_brightness")[18:]
    predicted_pupil_sizes = np.array( [predict_function(l) for l in brightnesses] )
    difference, arousal = calculate(pupil_sizes, brightnesses, predict_function)

    print("len(increments):", len(increments))
    print("len(predicted_pupil_sizes):", len(predicted_pupil_sizes))
    print("len(pupil_sizes):", len(pupil_sizes))
    print("len(brightnesses):", len(brightnesses))
    print("len(arousal):", len(arousal))

    # --- Plot calculation ---
    plt.figure()
    plt.plot(brightnesses, label="Brightness", linewidth=2)
    plt.plot(clean_pupil_sizes(pupil_sizes), label="Actual Pupil Size", linewidth=2)
    plt.plot(difference, label="Difference", linewidth=2)
    plt.plot(np.array( [predict_function(l) for l in brightnesses] ), label="Predicted Pupil Size", linewidth=2)
    plt.plot(arousal, label="Arousal", linewidth=2)
    plt.title("Brightness vs Pupil Size vs Predicted Pupil Size vs Arousal")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()