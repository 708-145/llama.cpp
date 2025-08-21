import math

def enumerate_fp4_values():
    """
    Enumerates all possible values for a simplified 4-bit floating-point format.

    The format is defined as:
    - 1 sign bit (S)
    - 2 exponent bits (E)
    - 1 mantissa bit (M)

    The value is calculated using the formula:
    (-1)^S * 2^(E - Bias) * (1.M)

    Here, we use a bias of 1, and the mantissa has an implicit leading 1.
    For example, a mantissa bit of 1 means the mantissa value is (1.1) in binary, which is 1.5 in decimal.
    """
    values = []
    
    # Iterate through all 16 possible 4-bit combinations
    for i in range(16):
        # Extract the bits using bitwise operations
        sign_bit = (i >> 3) & 1
        exponent_bits = (i >> 1) & 3
        mantissa_bit = i & 1
        
        # Determine the sign
        sign = -1 if sign_bit == 1 else 1
        
        # Calculate the exponent value, with a bias of 1
        exponent = exponent_bits - 1
        
        # Calculate the mantissa value. The leading '1.' is implicit.
        # The mantissa bit 'M' corresponds to a value of 0.5^1.
        mantissa = 1.0 + mantissa_bit * 0.5
        
        # Combine the parts to get the final floating-point value
        value = sign * (2 ** exponent) * mantissa
        values.append(value)
        
    return values

def main():
    """
    Main function to run the enumeration, sorting, and factor calculation.
    """
    # Get all 16 FP4 values
    all_values = enumerate_fp4_values()

    # Filter for positive, non-zero values for meaningful factor calculation
    positive_values = sorted([val for val in all_values if val > 0])
    
    # Check if there are enough values to compute factors
    if len(positive_values) < 2:
        print("Not enough positive values to compute factors.")
        return

    print("Enumerated and sorted positive FP4 values:")
    print("-" * 40)
    for val in positive_values:
        print(f"  {val:.4f}")

    print("\nFactor from one value to the next larger one:")
    print("-" * 40)
    factors = []
    for i in range(1, len(positive_values)):
        current_value = positive_values[i]
        previous_value = positive_values[i-1]
        
        # Calculate the factor. We use a try-except block to handle potential division by zero,
        # although in this specific model, it will not occur.
        try:
            factor = current_value / previous_value
            print(f"  {previous_value:.4f} -> {current_value:.4f}: Factor = {factor:.4f}")
            factors.append(factor)
        except ZeroDivisionError:
            print(f"  {previous_value:.4f} -> {current_value:.4f}: Division by zero skipped.")

    if factors:
        average_factor = sum(factors) / len(factors)
        print(f"\nAverage Factor: {average_factor:.4f}")

if __name__ == "__main__":
    main()

