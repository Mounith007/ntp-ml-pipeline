# generate_data.py
import numpy as np
import pandas as pd

def calculate_ntp_performance(power_mw, mass_flow_rate, expansion_ratio):
    """Calculates Thrust and Isp based on simplified physics."""
    # Constants
    R_universal = 8314.4  # J/(kmol·K)
    M_hydrogen = 2.016  # kg/kmol
    gamma = 1.405 # Specific heat ratio for hydrogen
    Cp_h2 = 14300 # Specific heat of H2 in J/(kg·K)
    g0 = 9.81 # Standard gravity

    # Calculate propellant temperature from reactor power
    propellant_temp = (power_mw * 1e6) / (mass_flow_rate * Cp_h2)

    # Calculate exhaust velocity (Ve) using the rocket equation
    term1 = (2 * gamma * R_universal) / ((gamma - 1) * M_hydrogen)
    term2 = propellant_temp * (1 - (1 / expansion_ratio)**((gamma - 1) / gamma))
    exhaust_velocity = np.sqrt(term1 * term2)

    # Calculate Thrust (F = ṁ * Ve)
    thrust = mass_flow_rate * exhaust_velocity

    # Calculate Specific Impulse (Isp = Ve / g0)
    isp = exhaust_velocity / g0

    return thrust, isp, propellant_temp

# --- Generate Data ---
def generate_dataset(num_samples=5000):
    data = []
    # Define realistic ranges for inputs
    power_range = [100, 1500] # MW
    flow_rate_range = [2, 45] # kg/s
    expansion_ratio_range = [100, 500]

    for _ in range(num_samples):
        power = np.random.uniform(*power_range)
        flow_rate = np.random.uniform(*flow_rate_range)
        expansion_ratio = np.random.uniform(*expansion_ratio_range)
        thrust, isp, temp = calculate_ntp_performance(power, flow_rate, expansion_ratio)
        data.append([power, flow_rate, expansion_ratio, thrust, isp, temp])

    columns = ['Power (MW)', 'Flow Rate (kg/s)', 'Expansion Ratio', 'Thrust (N)', 'Isp (s)', 'Temperature (K)']
    df = pd.DataFrame(data, columns=columns)
    df.to_csv('ntp_synthetic_data.csv', index=False)
    print(f"Successfully generated ntp_synthetic_data.csv with {num_samples} samples.")

if __name__ == "__main__":
    generate_dataset()