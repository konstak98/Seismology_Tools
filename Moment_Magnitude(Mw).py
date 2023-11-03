import obspy
from obspy import read
from obspy.signal.invsim import cosine_taper
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.offline as pyo

def process_sac_file():
    # Read the SAC file containing the velocity waveform
    st = read('SAC/1997.02.04-10.37.37.ALN.1.sac', format='SAC')
    print(st)
    st.plot()

    # Sample rate in Hz
    sampling_rate = st[0].stats.sampling_rate

    # Integrate the velocity waveform to obtain displacement (offset)
    st.integrate(method='cumtrapz')

    # Apply a cosine taper to reduce edge effects (optional but recommended)
    taper_percentage = 0.05  # Adjust as needed
    st.detrend('linear')
    st.taper(max_percentage=taper_percentage, type='cosine')

    # Save the integrated waveform as a new SAC file (optional)
    st.write("Integrated_SAC/integrated_1997.02.04-10.37.37.ALN.1.sac", format="SAC")

    st1 = read('Integrated_SAC/integrated_1997.02.04-10.37.37.ALN.1.sac', format="SAC")
    print(st1)

    # Plot the integrated displacement waveform
    st1.plot()

    # Access the integrated displacement data and start time
    trace = st[0]
    data = trace.data
    starttime = trace.stats.starttime

    # Calculate the FFT (Fast Fourier Transform) of the data
    spectrum = np.abs(np.fft.fft(st[0].data))

    # Create the corresponding frequency array
    freqs = np.fft.fftfreq(len(spectrum), 1.0 / sampling_rate)

    # Calculate the displacement spectrum in meters
    displacement_spectrum = 2.0 * np.abs(spectrum)

    # Convert to log scale
    log_displacement_spectrum = np.log10(displacement_spectrum)
    log_freqs = np.log10(freqs)

    # Plot the displacement spectrum with log-log axes
    plt.figure()
    plt.loglog(10 ** log_freqs, 10 ** log_displacement_spectrum, color='blue')
    plt.xlabel('Log(Frequency) (Hz)')
    plt.ylabel('Log(Displacement) (meters)')
    plt.title('Displacement Spectrum')
    plt.grid(True)
    plt.show()

    # Create a Plotly scatter plot with logarithmic axes
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=10 ** log_freqs, y=10 ** log_displacement_spectrum, mode='lines', line=dict(color='blue')))
    fig.update_xaxes(type='log', title='Log(Frequency) (Hz)')
    fig.update_yaxes(type='log', title='Log(Displacement) (meters)')
    fig.update_layout(title='Displacement Spectrum', showlegend=False)
    # Show the plot
    pyo.iplot(fig)

    # Calculate the angular frequency Fc (frequency at peak spectral amplitude)
    peak_index = np.argmax(log_displacement_spectrum)  # Find the index of the peak
    Fc = 10 ** log_freqs[peak_index]  # Convert back to linear frequency

    # Calculate the spectral amplitude C0 (peak spectral amplitude)
    C0 = 10 ** log_displacement_spectrum[peak_index]  # Convert back to linear amplitude

    print("Angular Frequency Fc:", Fc, "Hz")
    print("Spectral Amplitude C0:", C0, "meters")

    # Create a Plotly scatter plot with logarithmic axes
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=10 ** log_freqs, y=10 ** log_displacement_spectrum, mode='lines', line=dict(color='blue')))
    # Add markers for Fc and C0
    fig.add_trace(go.Scatter(x=[Fc], y=[C0], mode='markers', marker=dict(size=10, color='red'), name='Fc and C0'))
    # Add dotted lines connecting the peak point to the axes
    fig.add_trace(go.Scatter(x=[Fc, Fc], y=[0, C0], mode='lines+markers', marker=dict(size=1, color='green', symbol='line-ns-open')))
    fig.add_trace(go.Scatter(x=[0, Fc], y=[C0, C0], mode='lines+markers', marker=dict(size=1, color='purple', symbol='line-ew-open')))
    fig.update_xaxes(type='log', title='Log(Frequency) (Hz)')
    fig.update_yaxes(type='log', title='Log(Displacement) (meters)')
    fig.update_layout(title='Log-Log Displacement Spectrum with Fc and C0', showlegend=True)
    fig.show()

    # Access the SAC header and extract the hypocentral distance (R)
    header = st[0].stats.sac
    hypocentral_distance_meters = header.dist  # Hypocentral distance in meters

    # Given constants
    rock_density_gr_cm3 = 3.5  # g/cm³
    S_wave_velocity_km_sec = 4.5  # km/sec
    spectral_amplitude_meters = C0  # You have previously calculated this

    # Convert units to SI units
    rock_density_kg_m3 = rock_density_gr_cm3 * 1000  # g/cm³ to kg/m³
    S_wave_velocity_m_sec = S_wave_velocity_km_sec * 1000  # km/sec to m/sec

    # Calculate the seismic moment Mo
    Mo = (4 * np.pi * rock_density_kg_m3 * S_wave_velocity_m_sec**3 * spectral_amplitude_meters * hypocentral_distance_meters) / 0.85

    # Convert Mo to gr·cm²·sec⁻²
    Mo_gr_cm2_sec2 = Mo  # Since 1 dyn·cm = 1 gr·cm²·sec⁻²

    print("Seismic Moment Mo:", Mo_gr_cm2_sec2, "gr·cm²·sec⁻²")

    # Given constants
    Vs = 4.5  # Velocity of transverse waves in km/s
    Ks = 1.38  # Model constant for S waves

    # Calculate the radius (r)
    r = (Vs - Ks) / (2 * np.pi * Fc)

    # Calculate the area (A) of the circular bursting surface
    A = np.pi * r**2

    print("Radius (r):", r, "meters")
    print("Area (A) of the circular bursting surface:", A, "square meters")

    # Calculate the seismic moment magnitude (Mw)
    logMo = np.log10(Mo)  # Take the logarithm of Mo (in gr·cm²·sec⁻²)
    Mw = (2/3) * logMo - 10.73

    print("Seismic Moment Magnitude (Mw):", Mw)

    # Frequency of interest (0.05 Hz corresponds to a period of 20 seconds)
    target_frequency = 0.05  # Hz

    # Find the index in the frequency array that corresponds to the target frequency
    # You can use your existing frequency array and find the nearest frequency value
    # This index represents the frequency closest to the target_frequency
    index = np.argmin(np.abs(freqs - target_frequency))

    # Extract the amplitude at the target frequency from the displacement spectrum
    amplitude_micro_meters = 10 ** log_displacement_spectrum[index]  # Convert back to linear amplitude

    # Given sub-central distance in kilometers (R)
    # Access the SAC header and extract the hypocentral distance (R)
    header = st[0].stats.sac
    hypocentral_distance_meters = header.dist  # Hypocentral distance in meters
    subcentral_distance_km = hypocentral_distance_meters  # Replace with the actual value from the header

    # Calculate the surface magnitude (Ms)
    Ms = np.log10(amplitude_micro_meters) + 1.41 * np.log10(subcentral_distance_km) + 0.2

    print("Surface Magnitude (Ms):", Ms)

    # Given constants
    Q_D_h = 5.2  # Focal-point path parameter
    T = 1.0  # Period in seconds

    # Measure the maximum amplitude of the longitudinal waves in the first 20-30 seconds
    # Let's assume you have measured this amplitude and stored it in 'amplitude_micro_meters' in microns (μm)

    # Calculate the spatial magnitude (mb)
    mb = np.log10(amplitude_micro_meters / T) + Q_D_h

    print("Spatial Magnitude (mb):", mb)

if __name__ == '__main__':
    process_sac_file()
