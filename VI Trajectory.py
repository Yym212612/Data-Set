import os
import numpy as np
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt


def normalize_data(column_data):
    return (column_data - column_data.min()) / (column_data.max() - column_data.min())


def compute_power(voltage, current):
    return voltage * current


def compute_power_factor(voltage, current):
    P_active = np.mean(voltage * current)
    V_rms = np.sqrt(np.mean(voltage**2))
    I_rms = np.sqrt(np.mean(current**2))
    S_apparent = V_rms * I_rms
    PF = P_active / S_apparent
    return PF


def compute_hue(voltage, current):
    H = np.zeros_like(voltage)
    for j in range(len(voltage) - 1):

        delta_v = voltage[j + 1] - voltage[j]
        delta_i = current[j + 1] - current[j]

        angle = np.arctan2(delta_v, delta_i)

        normalized_angle = (angle + np.pi) / (2 * np.pi)
        H[j] = normalized_angle
    H[-1] = H[-2]
    return H


def compute_third_harmonic(data, sample_rate):

    fft_result = np.fft.fft(data)

    freqs = np.fft.fftfreq(len(data), d=1/sample_rate)


    third_harmonic_freq = 3 * 60

    third_harmonic_index = np.argmin(np.abs(freqs - third_harmonic_freq))

    third_harmonic_magnitude = np.abs(fft_result[third_harmonic_index])
    return third_harmonic_magnitude


def plot_3d_power_hue_vi_trajectory(data, save_path, sample_rate):
    voltage = data.iloc[:, 1].values
    current = data.iloc[:, 0].values
    time_interval = 16.65 / len(voltage)
    time_stamps = np.arange(0, 16.65, time_interval)

    PF = compute_power_factor(voltage, current)
    H = compute_hue(voltage, current)
    S = np.ones_like(H) * (0.5 + 0.5 * PF)
    V_value = compute_third_harmonic(voltage, sample_rate) / np.max(compute_third_harmonic(voltage, sample_rate))
    V = np.full_like(H, V_value)

    HSV = np.stack((H, S, V), axis=-1)
    RGB = colors.hsv_to_rgb(HSV)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    for i in range(1, len(voltage)):
        ax.plot(voltage[i - 1:i + 1], current[i - 1:i + 1], time_stamps[i - 1:i + 1], color=RGB[i])

    ax.set_xlabel("Normalized Voltage/(V)")
    ax.set_ylabel("Normalized Current/(A)")
    ax.set_zlabel("Time (ms)")
    plt.savefig(save_path)
    plt.close()

def sort_key(filename):

    numbers = [int(s) for s in filename.split('.') if s.isdigit()]
    return numbers[0] if numbers else -1


csv_folder_path = r'C:/Users/Administrator/Desktop/yuanyimin/PLAID'


base_save_folder_path = r'C:/Users/Administrator/Desktop/yuanyimin/Colorful VI Trajectory'


sample_rate = 30000


all_csv_files = [f for f in os.listdir(csv_folder_path) if f.endswith('.csv')]
sorted_csv_files = sorted(all_csv_files, key=lambda f: int(''.join(filter(str.isdigit, f))))


for idx, filename in enumerate(sorted_csv_files):

    folder_number = int(''.join(filter(str.isdigit, filename)))
    print(f"Processing file {idx + 1}/{len(sorted_csv_files)}: {filename} into folder {folder_number}")

    try:

            save_folder_path = os.path.join(base_save_folder_path, str(folder_number))
            os.makedirs(save_folder_path, exist_ok=True)


            csv_file_path = os.path.join(csv_folder_path, filename)
            data = pd.read_csv(csv_file_path, header=None)


            data.iloc[:, 0] = normalize_data(data.iloc[:, 0])
            data.iloc[:, 1] = normalize_data(data.iloc[:, 1])


            samples_per_cycle = 500
            number_of_cycles = len(data) // samples_per_cycle

            for i in range(number_of_cycles):
                start = i * samples_per_cycle
                end = (i + 1) * samples_per_cycle
                cycle_data = data.iloc[start:end, :]
                save_path = os.path.join(save_folder_path, f"VI Trajectory {i + 1}.png")
                plot_3d_power_hue_vi_trajectory(cycle_data, save_path, sample_rate)

            print(f"Finished processing {filename} into folder {folder_number}")

    except Exception as e:
            print(f"An error occurred while processing {filename}: {e}")
