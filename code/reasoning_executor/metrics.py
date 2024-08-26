from fractions import Fraction
import re

def compare_results(output, ground_truth):
    # Normalize strings to lower case
    output = output.strip().lower()
    ground_truth = ground_truth.strip().lower()

    # Direct string comparison for exact match
    if output == ground_truth:
        return True

    # Handle True/False and Yes/No equivalence
    if ground_truth in ["yes", "no"]:
        if output in ["true", "yes"]:
            return ground_truth == "yes"
        elif output in ["false", "no"]:
            return ground_truth == "no"
        else:
            return False

    # Check for substring match in case of non-numeric values
    if not re.search(r'\d', output) and not re.search(r'\d', ground_truth):
        return output in ground_truth or ground_truth in output

    # Extract numeric values from strings with additional text
    output_numeric = re.sub(r'[^0-9./-]', '', output)
    ground_truth_numeric = re.sub(r'[^0-9./-]', '', ground_truth)

    # Direct string comparison for non-numeric values
    if not output_numeric or not ground_truth_numeric:
        return output == ground_truth

    try:
        # Convert to float for comparison
        output_value = float(Fraction(output_numeric))
        ground_truth_value = float(Fraction(ground_truth_numeric))

        # Round numbers to 5 decimal places
        decimal_places = 5
        output_rounded = round(output_value, decimal_places)
        ground_truth_rounded = round(ground_truth_value, decimal_places)

        return output_rounded == ground_truth_rounded
    except ValueError:
        # Handle cases where conversion to float fails
        return False