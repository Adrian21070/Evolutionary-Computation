import numpy as np

class Individual():

    def __init__(self, genome: list[float] |  str) -> None:
        
        self.genome = genome
        self.phenotype: list[float] | None = None

        self.fitness: float|None = None

    def set_phenotype(self, phenotype: list[float]) -> None:
        self.phenotype = phenotype

    def __repr__(self):
        return f"Individual(phenotype={self.phenotype}, fitness={self.fitness})"
    

class Codec():

    def __init__(self, kind: str, precision: int, intervals: list[float, float]) -> None:

        self.genome_kind = kind.lower()
        self.precision = precision

        # Compute the bit length
        self.scale_factor = 10 ** self.precision
        self.bit_length = int(np.log2((intervals[1] - intervals[0]) * self.scale_factor) + 0.99)

    def encode(self, real_values: list[float]) -> list | str:
        """
        
        """
        if self.genome_kind == "binary":
            # Conver each float into int
            int_values: list[int] = [int(abs(value * self.scale_factor)) for value in real_values]
            sign_values: list[bool] = np.sign(real_values)
        
            # Encode each integer to n-bit length string
            genome: list[str] = [f"{value:0{self.bit_length}b}" for value in int_values]

            # Append sign bit at the beggining of each string
            for i in range(len(sign_values)):
                if sign_values[i] == 1:
                    genome[i] = "0" + genome[i]
                else:
                    genome[i] = "1" + genome[i]

            return "".join(genome)
        
        elif self.genome_kind == "real":
            return real_values

    def decode(self, genome: list | str):
        
        if self.genome_kind == "binary":
            # Convert from binary to real values
            real_values = []

            # Iterate over the genome with steps of self.bit_length + 1 to include sign
            for i in range(0, len(genome), self.bit_length + 1):
                
                # Get chunk of len bit_length
                binary_str = genome[i: i + self.bit_length + 1]

                # Get sign
                sign = 1 if binary_str[0] == "0" else -1

                # From binary to integer
                integer_value = int(binary_str[1:], 2)
                
                # Downscale by scale factor
                real_value = sign * integer_value / self.scale_factor    
                real_values.append(real_value)
            
            return real_values
        
        elif self.genome_kind == "real":
            return genome

class Population():

    def __init__(self, pop_size: int) -> None:
    
        self.pop_size = pop_size
        self.population: list[Individual] = []
    

    def initialize(self, num_variables: int, intervals: tuple[float, float], codec: Codec) -> None:
        
        # Random initialization
        interval = intervals[1] - intervals[0]
        for _ in range(self.pop_size):

            # Make a list of random numbers
            random_list: list[float] = (np.random.rand(num_variables) * interval + intervals[0]).tolist()

            # Encode the random list to get the genome
            genome: list[float] | str = codec.encode(random_list)

            # Create an individual
            individual = Individual(genome)

            # Set its phenotype
            individual.set_phenotype(random_list)

            # Add it to the population
            self.population.append(individual)

    def get_population(self) -> list[Individual]:
        return self.population

    def update_population(self, new_population: list[Individual]) -> None:
        self.population = new_population

    def get_fitness(self) -> list[float]:
        fitness_values: list[float] = []
        for individual in self.population:
            fitness_values.append(individual.fitness)
        
        return fitness_values
