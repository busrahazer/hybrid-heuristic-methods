import numpy as np
from datetime import datetime

# ==================== PROBLEM SINIFI ====================
class PMSProblem:

    def __init__(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        
        self.T = 52  # Zaman horizonu (52 hafta)
        self.n_units = 8
        self.n_boilers = 8
        self.n_turbines = 8
        self.n_distillers = 16
        self.total_equipment = self.n_boilers + self.n_turbines + self.n_distillers

        # Bakım süreleri (hafta cinsinden)
        self.maintenance_durations = {
            'boiler': 5,
            'turbine': 4,
            'distiller': 5
        }

        # Üretim kapasiteleri
        self.production_capacity = {
            'turbine': 47040,  # MW
            'distiller': 50.4   # MIGD
        }

        # Talep verileri
        self.water_demand = self.generate_demand('water')
        self.electricity_demand = self.generate_demand('electricity')

        # Kısıtlar
        self.max_concurrent_maintenances = 2
        self.workforce_limit = 100

    def generate_demand(self, type_):
        """Talep verisi üretme (yaz aylarında artış)"""
        base_demand = np.ones(self.T)
        summer_weeks = range(20, 32)

        if type_ == 'water':
            base_demand *= 300  # MIGD
            base_demand[summer_weeks] *= 1.3
        else:  # electricity
            base_demand *= 300000  # MW
            base_demand[summer_weeks] *= 1.25

        # Rastgele dalgalanma
        noise = np.random.normal(0, 0.05, self.T)
        base_demand = base_demand * (1 + noise)

        return base_demand
    
# ==================== GENETİK ALGORİTMA ====================
class GeneticAlgorithm:

    def __init__(self, problem, pop_size=80, generations=100, 
                 crossover_rate=0.8, mutation_rate=0.1, verbose=True):
        self.problem = problem
        self.pop_size = pop_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.verbose = verbose

        # İstatistikler
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.diversity_history = []  # Popülasyon çeşitliliği
        self.improvement_per_generation = []  # Her nesilde iyileşme

    def create_chromosome(self):
        """Geçerli bir kromozom oluştur"""
        chromosome = np.zeros(self.problem.T, dtype=int)
        
        for eq_id in range(1, self.problem.total_equipment + 1):
            eq_type = self.get_equipment_type(eq_id)
            duration = self.problem.maintenance_durations[eq_type]

            for _ in range(100):
                start_week = np.random.randint(0, self.problem.T - duration + 1)
                
                if all(chromosome[start_week + i] == 0 for i in range(duration)):
                    for i in range(duration):
                        chromosome[start_week + i] = eq_id
                    break

        return chromosome

    def get_equipment_type(self, eq_id):
        """Ekipman tipini belirle"""
        if eq_id <= self.problem.n_boilers:
            return 'boiler'
        elif eq_id <= self.problem.n_boilers + self.problem.n_turbines:
            return 'turbine'
        else:
            return 'distiller'

    def calculate_fitness(self, chromosome):
        """Fitness hesaplama + detaylı sonuçlar"""
        water_production = np.zeros(self.problem.T)
        electricity_production = np.zeros(self.problem.T)

        for week in range(self.problem.T):
            equipment_in_maintenance = chromosome[week]
            active_turbines = self.problem.n_turbines
            active_distillers = self.problem.n_distillers

            if equipment_in_maintenance != 0:
                eq_type = self.get_equipment_type(equipment_in_maintenance)
                if eq_type == 'turbine':
                    active_turbines -= 1
                elif eq_type == 'distiller':
                    active_distillers -= 1
                elif eq_type == 'boiler':
                    active_turbines -= 1
                    active_distillers -= 2

            electricity_production[week] = active_turbines * self.problem.production_capacity['turbine']
            water_production[week] = active_distillers * self.problem.production_capacity['distiller']

        water_gap = water_production - self.problem.water_demand
        electricity_gap = electricity_production - self.problem.electricity_demand

        penalty = 0
        penalty += np.sum(np.abs(water_gap[water_gap < 0])) * 1000
        penalty += np.sum(np.abs(electricity_gap[electricity_gap < 0])) * 1000

        fitness = np.std(water_gap) + np.std(electricity_gap) + penalty

        # Detaylı sonuçlar
        details = {
            'fitness': fitness,
            'water_gap': water_gap,
            'electricity_gap': electricity_gap,
            'water_std': np.std(water_gap),
            'electricity_std': np.std(electricity_gap),
            'penalty': penalty,
            'min_water_gap': np.min(water_gap),
            'min_electricity_gap': np.min(electricity_gap)
        }

        return fitness, details

    def calculate_diversity(self, population):
        """Popülasyon çeşitliliğini hesapla"""
        if len(population) < 2:
            return 0
        
        # Ortalama Hamming distance
        total_distance = 0
        comparisons = 0
        
        for i in range(len(population)):
            for j in range(i+1, min(i+10, len(population))):  # Her bireyi 10 diğeriyle karşılaştır
                distance = np.sum(population[i] != population[j])
                total_distance += distance
                comparisons += 1
        
        return total_distance / comparisons if comparisons > 0 else 0

    def selection(self, population, fitness_scores):
        tournament_size = 5
        selected = []

        for _ in range(len(population)):
            tournament_idx = np.random.choice(len(population), tournament_size, replace=False)
            tournament_fitness = [fitness_scores[i] for i in tournament_idx]
            winner_idx = tournament_idx[np.argmin(tournament_fitness)]
            selected.append(population[winner_idx].copy())

        return selected

    def crossover(self, parent1, parent2):
        """Tek noktalı çaprazlama"""
        if np.random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()

        point = np.random.randint(1, len(parent1))
        child1 = np.concatenate([parent1[:point], parent2[point:]])
        child2 = np.concatenate([parent2[:point], parent1[point:]])

        return child1, child2

    def mutate(self, chromosome):
        """Swap mutation"""
        if np.random.random() > self.mutation_rate:
            return chromosome

        mutated = chromosome.copy()
        idx1, idx2 = np.random.choice(len(chromosome), 2, replace=False)
        mutated[idx1], mutated[idx2] = mutated[idx2], mutated[idx1]

        return mutated

    def run(self):
        """GA'yı çalıştır"""
        start_time = datetime.now()

        # Başlangıç popülasyonu
        population = [self.create_chromosome() for _ in range(self.pop_size)]

        best_solution = None
        best_fitness = float('inf')
        previous_best = float('inf')

        for gen in range(self.generations):
            # Fitness hesapla
            fitness_scores = []
            for chrom in population:
                fit, _ = self.calculate_fitness(chrom)
                fitness_scores.append(fit)

            # En iyi çözümü güncelle
            gen_best_idx = np.argmin(fitness_scores)
            gen_best_fitness = fitness_scores[gen_best_idx]

            if gen_best_fitness < best_fitness:
                best_fitness = gen_best_fitness
                best_solution = population[gen_best_idx].copy()

            # İstatistikleri kaydet
            self.best_fitness_history.append(best_fitness)
            self.avg_fitness_history.append(np.mean(fitness_scores))
            self.diversity_history.append(self.calculate_diversity(population))
            
            # İyileşme miktarı
            improvement = previous_best - best_fitness
            self.improvement_per_generation.append(improvement)
            previous_best = best_fitness

            # Yeni nesil oluştur
            selected = self.selection(population, fitness_scores)

            new_population = []
            for i in range(0, len(selected), 2):
                if i + 1 < len(selected):
                    child1, child2 = self.crossover(selected[i], selected[i+1])
                    child1 = self.mutate(child1)
                    child2 = self.mutate(child2)
                    new_population.extend([child1, child2])

            population = new_population[:self.pop_size]

            if self.verbose and gen % 50 == 0:
                print(f"Nesil {gen}: En İyi = {best_fitness:.2f}, Ortalama = {np.mean(fitness_scores):.2f}, Çeşitlilik = {self.diversity_history[-1]:.1f}")

        end_time = datetime.now()
        computation_time = (end_time - start_time).total_seconds()

        if self.verbose:
            print(f"✓ GA tamamlandı: Fitness = {best_fitness:.2f}, Süre = {computation_time:.2f}s")

        # Final detaylı sonuçlar
        _, final_details = self.calculate_fitness(best_solution)

        return best_solution, best_fitness, computation_time, final_details    
    
# ==================== TABU SEARCH ====================
class TabuSearch:

    def __init__(self, problem, tabu_tenure=4, max_iterations=50, verbose=True):
        self.problem = problem
        self.tabu_tenure = tabu_tenure
        self.max_iterations = max_iterations
        self.verbose = verbose
        
        self.tabu_list = []
        self.best_fitness_history = []
        self.accepted_moves = []  # Kabul edilen hamle sayısı
        self.aspiration_triggers = []  # Aspiration criteria kaç kez tetiklendi

    def calculate_fitness(self, chromosome):
        """Fitness hesaplama"""
        water_production = np.zeros(self.problem.T)
        electricity_production = np.zeros(self.problem.T)

        for week in range(self.problem.T):
            equipment_in_maintenance = chromosome[week]
            active_turbines = self.problem.n_turbines
            active_distillers = self.problem.n_distillers

            if equipment_in_maintenance != 0:
                eq_type = self.get_equipment_type(equipment_in_maintenance)
                if eq_type == 'turbine':
                    active_turbines -= 1
                elif eq_type == 'distiller':
                    active_distillers -= 1
                elif eq_type == 'boiler':
                    active_turbines -= 1
                    active_distillers -= 2

            electricity_production[week] = active_turbines * self.problem.production_capacity['turbine']
            water_production[week] = active_distillers * self.problem.production_capacity['distiller']

        water_gap = water_production - self.problem.water_demand
        electricity_gap = electricity_production - self.problem.electricity_demand

        penalty = 0
        penalty += np.sum(np.abs(water_gap[water_gap < 0])) * 1000
        penalty += np.sum(np.abs(electricity_gap[electricity_gap < 0])) * 1000

        fitness = np.std(water_gap) + np.std(electricity_gap) + penalty

        return fitness

    def get_equipment_type(self, eq_id):
        if eq_id <= self.problem.n_boilers:
            return 'boiler'
        elif eq_id <= self.problem.n_boilers + self.problem.n_turbines:
            return 'turbine'
        else:
            return 'distiller'

    def generate_neighbors(self, solution):
        """Komşu çözümler üret - SWAP operasyonu"""
        neighbors = []

        for i in range(len(solution)):
            for j in range(i+1, min(i+10, len(solution))):
                if (i, j) not in self.tabu_list:
                    neighbor = solution.copy()
                    neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
                    neighbors.append((neighbor, (i, j), 'swap'))

        return neighbors

    def update_tabu_list(self, move):
        """Tabu listesini güncelle"""
        self.tabu_list.append(move)
        if len(self.tabu_list) > self.tabu_tenure:
            self.tabu_list.pop(0)

    def run(self, initial_solution):
        """Tabu Search'ü çalıştır"""
        if self.verbose:
            print("\n--- Tabu Search Başlatılıyor ---")
        
        start_time = datetime.now()

        current_solution = initial_solution.copy()
        current_fitness = self.calculate_fitness(current_solution)

        best_solution = current_solution.copy()
        best_fitness = current_fitness

        self.best_fitness_history.append(best_fitness)

        iterations_without_improvement = 0
        aspiration_count = 0

        for iteration in range(self.max_iterations):
            neighbors = self.generate_neighbors(current_solution)

            if not neighbors:
                break

            best_neighbor = None
            best_neighbor_fitness = float('inf')
            best_move = None
            aspiration_used = False

            for neighbor, move, move_type in neighbors:
                neighbor_fitness = self.calculate_fitness(neighbor)

                # Aspiration criteria
                if neighbor_fitness < best_fitness:
                    best_neighbor = neighbor
                    best_neighbor_fitness = neighbor_fitness
                    best_move = move
                    aspiration_used = True
                    aspiration_count += 1
                    break

                if neighbor_fitness < best_neighbor_fitness:
                    best_neighbor = neighbor
                    best_neighbor_fitness = neighbor_fitness
                    best_move = move

            current_solution = best_neighbor
            current_fitness = best_neighbor_fitness

            if best_move:
                self.update_tabu_list(best_move)
                self.accepted_moves.append(1)
            else:
                self.accepted_moves.append(0)

            if current_fitness < best_fitness:
                best_solution = current_solution.copy()
                best_fitness = current_fitness
                iterations_without_improvement = 0
                if self.verbose:
                    aspiration_marker = " [ASPIRATION]" if aspiration_used else ""
                    print(f"TS İterasyon {iteration}: YENİ EN İYİ = {best_fitness:.2f}{aspiration_marker} ✓")
            else:
                iterations_without_improvement += 1

            self.best_fitness_history.append(best_fitness)
            self.aspiration_triggers.append(aspiration_count)

            if iterations_without_improvement > 15:
                break

        end_time = datetime.now()
        computation_time = (end_time - start_time).total_seconds()

        if self.verbose:
            print(f"✓ TS tamamlandı: Fitness = {best_fitness:.2f}, Süre = {computation_time:.2f}s")
            print(f"  Aspiration kullanım sayısı: {aspiration_count}")

        return best_solution, best_fitness, computation_time
