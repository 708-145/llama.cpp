import math

def enumerate_fp8_values():
    """
    Enumerates all possible values for a simplified 8-bit floating-point format.

    The format is defined as:
    - 1 sign bit (S)
    - 4 exponent bits (E)
    - 3 mantissa bits (M)

    The value is calculated using the formula:
    (-1)^S * 2^(E - Bias) * (1.M)

    For this 4-bit exponent, the bias is 2^(4-1) - 1 = 7.
    The mantissa has an implicit leading '1'.
    """
    values = []
    
    # Iterate through all 256 possible 8-bit combinations
    for i in range(256):
        # Extract the bits using bitwise operations
        sign_bit = (i >> 7) & 1
        exponent_bits = (i >> 3) & 15
        mantissa_bits = i & 7
        
        # Determine the sign
        sign = -1 if sign_bit == 1 else 1
        
        # Calculate the exponent value, with a bias of 7
        exponent = exponent_bits - 7
        
        # Calculate the mantissa value from the 3 mantissa bits.
        # The bits correspond to fractional values of 0.5, 0.25, and 0.125.
        m1 = (mantissa_bits >> 2) & 1
        m2 = (mantissa_bits >> 1) & 1
        m3 = mantissa_bits & 1
        mantissa_fraction = m1 * 0.5 + m2 * 0.25 + m3 * 0.125
        
        # Add the implicit leading 1
        mantissa = 1.0 + mantissa_fraction
        
        # Combine the parts to get the final floating-point value
        value = sign * (2 ** exponent) * mantissa
        values.append(value)
        
    return values

def main():
    """
    Main function to run the enumeration, sorting, and factor calculation.
    """
    # Get all 256 FP8 values
    all_values = enumerate_fp8_values()

    # Filter for positive, non-zero values for meaningful factor calculation
    positive_values = sorted([val for val in all_values if val > 0])
    
    # Check if there are enough values to compute factors
    if len(positive_values) < 2:
        print("Not enough positive values to compute factors.")
        return

    print("Enumerated and sorted positive FP8 values:")
    print("-" * 50)
    for val in positive_values:
        print(f"  {val:.4f}")

    print("\nFactor from one value to the next larger one:")
    print("-" * 50)
    factors = []
    for i in range(1, len(positive_values)):
        current_value = positive_values[i]
        previous_value = positive_values[i-1]
        
        # Calculate the factor. This will be greater than 1.
        try:
            factor = current_value / previous_value
            print(f"  {previous_value:.4f} -> {current_value:.4f}: Factor = {factor:.4f}")
            factors.append(factor)
        except ZeroDivisionError:
            # This case should not be reached with the current filtering
            pass

    if factors:
        average_factor = sum(factors) / len(factors)
        print(f"\nAverage Factor: {average_factor:.4f}")

if __name__ == "__main__":
    main()

