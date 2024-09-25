
import numpy as np

from typing import Union, Optional

class Codec():

    def __init__(self, kind: str, num_variables: int, precision: int,
                  constraint: list[float, float]) -> None:

        # Define encoding type
        self.binary = True if kind.lower() == "binary" else False

        # Save boundaries
        self.bounds = constraint

        # Define values for binary encoding
        if self.binary:

            # Set desired precision
            self.precision = precision

            # Compute the bit length
            self.compute_bit_length()

            # Genome length
            self.num_allels = num_variables * self.bit_length

        # Define values for real encoding
        else:
            # Genome length
            self.num_allels = num_variables
    
    def compute_bit_length(self) -> None:
        
        # Get interval from boundaries
        self.interval = self.bounds[1] - self.bounds[0]

        # Compute scale factor and bit length
        self.scale_factor = 10 ** self.precision

        # Compute number of bits to represent individuals
        self.bit_length = int(np.log2(self.interval * self.scale_factor) + 0.99)

        # Compute the max representative number
        self.max_representation = 2 ** self.bit_length - 1

    def encode(self, real_values: list[float]) -> Union[list[float], str]:
        """
        Parameters
        ----------
        real_values: list[float]
            Values to encode
        
        Returns
        -------
        list[float] | str
            Encoded genome
        """

        if self.binary:
            # Normalize each value into a [0,1] interval
            normalized: list[float] = [(x - self.bounds[0]) / self.interval for x in real_values]

            # Convert each normalized value into a int bounded by the number of bits
            int_values: list[int] = [int(value * self.max_representation) for value in normalized]

            # Encode each integer to n-bit length strings
            genome: list[str] = [f"{value:0{self.bit_length}b}" for value in int_values]

            return "".join(genome)

        else:
            # Real encoding
            return real_values
        
    def decode(self, genome: Union[list[float], str]) -> list[float]:
        """
        Parameters
        ----------
        genome: list[float] | str
            Genome to decode
        
        Returns
        -------
        list[float]
            Phenotype of individual
        """
        
        if self.binary:
            # Convert from binary to real values
            real_values = []

            # Iterate over the genome with steps of self.bit_length
            for i in range(0, len(genome), self.bit_length):

                # Get chunk of num of bits
                binary_str = genome[i: i + self.bit_length]

                # From binary to integer
                int_value = int(binary_str, 2)

                # Downscale by maximum representation
                y = int_value / self.max_representation

                # Scale value into boundaries
                value = y * self.interval + self.bounds[0]

                # Append value
                real_values.append(value)

            return real_values

        else:
            # Real values
            return genome
        

"""
if __name__ == "__main__":
    coder = Codec(kind="binary", num_variables=3, precision=4, constraint=[-5.12,5.12])

    genome = coder.encode([4.12, -2.14, 3.5])
    print(coder.decode(genome))

    # What happens when we are out of scope?
    # * Note: It doesn't respect the fixed bit length.
    
    genome = coder.encode([6])
    print(coder.decode(genome))
"""