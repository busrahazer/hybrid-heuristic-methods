import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

# Problem paramtetreleri 
class PMSProblem:
    """Preventive Maintenance Scheduling Problem"""

    def __init__(self):
        # Zaman horizonu (52 hafta)
        self.T = 52
        
        # Ekipman sayıları (Al-Zour tesisi - 8 ünite)
        self.n_units = 8
        self.n_boilers = 8
        self.n_turbines = 8
        self.n_distillers = 16 # Her ünitede 2tane damıtıcı var.

        # Toplam ekipman sayısı
        self.total_equipment = self.n_boilers + self.n_turbines + self.n_distillers

        # Bakım süreleri (hafta cinsinden)
        self.maintanance_durations = {
            'boiler': 5,
            'turbine': 4,
            'distiller': 5
        }

        # Üretim kapasiteleri
        self.production_capacity = {
            'turbine': 47040, # MWh/hafta
            'distiller': 50.4 # MGID (ünite 1-6), 40.2 MGID (ünite 7-8) / hafta
        }

        # Talep verileri (yaz aylarında artış)
        self.water_demand = self.generate_demand('water')
        self.electricity_demand = self.generate_demand('electricity')

        # Kısıtlar
        self.max_concurrent_maintenances = 2 # Aynı anda en fazla 2 ünite bakımda olabilir
        self.workforce_limit = 100 # Maksimum iş gücü

    def generate_demand(self, type):
        """Talep verisi üretme (yaz aylarında artış)"""
        base_demand = np.ones(self.T)
        # Yaz aylarında (hafta 21-32) artış
        summer_weeks =  range(20, 32)

        if type  == 'water':
            base_demand *=300 # MGID
            base_demand[summer_weeks] *= 1.3
        else: # elektrik
            base_demand *= 300000 # MW
            base_demand[summer_weeks] *= 1.25    

        # Rastgele dalgalanma eklenebilir
        noise = np.random.normal(0, 0.05, self.T)
        base_demand = base_demand * (1 + noise)

        return base_demand    

class GeneticAlgorithm:    
        """PMS için Genetik Algoritma"""

        def __init__(self, problem, pop_size=100, generations=200, crossover_rate=0.8, mutation_rate=0.2):
            self.problem = problem
            self.pop_size = pop_size
            self.generations = generations
            self.crossover_rate = crossover_rate
            self.mutation_rate = mutation_rate

            # İstatistikler
            self.best_fitness_history = []
            self.avg_fitness_history = []
            self.computation_times = []

        def create_chromosome(self):
            """
            Kromozom = 52 haftalık bakım çizelgesi
            Her gen = o hafta bakıma alınan ekipman ID'si (veya 0 = boş)
            """
            chromosome = np.zeros(self.problem.T, dtype=int)
            equipment_scheduled = set()

            # Her ekipman için bir başlangıç bakımı atama
            for eq_id in range(1, self.problem.total_equipment + 1):
                # Rastegele bir başlangıç haftası seç
                eq_type = self.get_equipment_type(eq_id)
                duration = self.problem.maintanance_durations[eq_type]

                # Geçerli bir hafta bul
                max_attempts = 100
                for i in range(max_attempts):
                    start_week = np.random.randint(0, self.problem.T - duration + 1)

                    # Bu haftalarda başka ekipman var mı?
                    if all(chromosome[start_week + i] == 0 for i in range (duration)):
                        # Bakımı ata
                        for i in range(duration):
                            chromosome[start_week + i] = eq_id
                        equipment_scheduled.add(eq_id)
                        break  

            return chromosome 

        def get_equipment_type(self, eq_id):
            """Ekipman ID'sine göre tip belirle"""
            if eq_id <= self.problem.n_boilers:
                return 'boiler'
            elif eq_id <= self.problem.n_boilers + self.problem.n_turbines:
                return 'turbine'
            else:
                return 'distiller'  

        def calculate_fitness(self, chromosome):
            """
            Fitness = üretim-talep farkının standart sapmasını minimize et
            Daha düşük fitness = daha iyi çözüm
            """   
            water_production = np.zeros(self.problem.T)
            electrity_production = np.zeros(self.problem.T) 

            # Her hafta için üretim kapasitesi hesaplama
            for week in range(self.problem.T):
                # Bakımda olmayan ekipmanları say
                equipment_in_maintanance = chromosome[week]       

                # Basitleşitirlmiş hesaplama
                # Gerçekte kazan (boiler) bakımda ise ona bağlı türbin ve damıtıcı da çalışmaz
                active_turbines = self.problem.n_turbines
                active_distilers = self.problem.n_distillers

                # Bakımdaki ekipmanların sayısını çıkart
                if equipment_in_maintanance != 0:
                    eq_type = self.get_equipment_type(equipment_in_maintanance)
                    if eq_type == 'turbine':
                        active_turbines -= 1
                    elif eq_type == 'distiller':
                        active_distilers -= 1
                        # Boiler bakımda ise ona bağlı türbin de damıtıcı da çalışmaz
                        active_turbines -= 1
                        active_distillers -= 2 # 2 damıtıcı olma sebebi; ünitelerde 1 boiler, 1 türbin ve 2 damıtıcı var

                # Üretimi hesapla
                electrity_production[week] = active_turbines * self.problem.production_capacity['turbine']
                water_production[week] = active_distilers * self.problem.production_capacity['distiller']  

            # Talep ile üretim farkı
            water_gap = water_production - self.problem.water_demand
            electricity_gap = electrity_production - self.problem.electricity_demand

            # Negatif gap = talebi karşılayamama (çok kötü), pozitif gap = fazla üretim    
            penalty = 0
            penalty += np.sum(np.abs(water_gap[water_gap <0])) * 1000 # büyük ceza yedi xd
            penalty += np.sum(np.abs(electricity_gap[electricity_gap <0])) * 1000

            # Fitness = standart sapma + ceza
            fitness = np.std(water_gap) + np.std(electricity_gap) + penalty

            return fitness, water_gap, electricity_gap
        
        def selection(self, population, fitness_scores):
            """Turnuva seçimi"""
            torunament_size = 5
            selected = []

            for i in range(len(population)):
                touranment_index = np.random.choice(len(population), torunament_size, replace=False)
                touranment_fitness = [fitness_scores[i] for i in touranment_index]
                winner_index = touranment_index[np.argmin(touranment_fitness)]
                selected.append(population[winner_index].copy())
                                
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
          """Swap mutation: iki rastgele hafta yer değiştirir"""
          if np.random.random() > self.mutation_rate:
              return chromosome

          mutated = chromosome.copy()
          index1, index2 = np.random.choice(len(chromosome), 2, replace=False)
          mutated[index1], mutated[index2] = mutated[index2], mutated[index1]

          return mutated    

        def run(self):
            """"GA'yı çalıştır"""
            start_time = datetime.now()

            # Başlangıç popülasyonu
            population = [self.create_chromosome() for _ in range(self.pop_size)]

            best_solution = None
            best_fitness = float('inf') 

            for gen in range(self.generations):
                # Fitness hesapla
                fitness_scores = []
                for chrom in population:
                    fit, _, _  = self.calculate_fitness(chrom)
                    fitness_scores.append(fit)

                # En iyi çözümü güncelle
                gen_best_index = np.argmin(fitness_scores)
                gen_best_fitness = fitness_scores[gen_best_index]

                if gen_best_fitness < best_fitness: 
                    best_fitness = gen_best_fitness
                    best_solution = population[gen_best_index].copy()

                # İstatistikleri kaydet
                self.best_fitness_history.append(best_fitness)  
                self.avg_fitness_history.append(np.mean(fitness_scores))

                # Yeni nesil oluştur
                selected = self.selection(population, fitness_scores)

                new_population = []
                for i in range(0, len(selected), 2):
                    if i + 1 < len(selected):
                        child1, child2 = self.crossover(selected[i], selected[i+1])
                        child1 = self.mutate(child1)
                        child2 = self.mutate(child2)
                        new_population.extend([child1, child2])

                population = new_population[:self.pop_size] # Popülasyon boyutunu koru

                if gen % 20 == 0:
                    print(f"Nesil {gen}: En İyi Fitness = {best_fitness:.2f}, Ortalama = {np.mean(fitness_scores):.2f}")    

            end_time = datetime.now()
            computation_time = (end_time - start_time).total_seconds()

            print(f"GA tamamlandı. En iyi fitness: {best_fitness:.2f}, Süre: {computation_time:.2f} saniye")

            return best_solution, best_fitness, computation_time
        
class TabuSearch:        
    """Tabu Search - GA'nın bulduğu çözümü iyileştirir"""
    def __init__(self, problem, tabu_tenure=4, max_iterations=50):
        self.problem = problem
        self.tabu_tenure = tabu_tenure # Tabu listede kalma süresi
        self.max_iterations = max_iterations
        self.tabu_list = []  # (hafta1, hafta2) formatında swap'ler
        self.best_fitness_history = []

    def calculate_fitness(self, chromosome):
        """Fitness hesaplama (GA ile aynı)"""
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
        """
        Komşuluk yapısı: SWAP ve INSERT operasyonları
        """
        neighbors = []
        
        # SWAP operasyonu: iki hafta yer değiştir
        for i in range(len(solution)):
            for j in range(i+1, min(i+10, len(solution))):  # Komşuluk boyutunu sınırla
                if (i, j) not in self.tabu_list:  # Tabu değilse
                    neighbor = solution.copy()
                    neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
                    neighbors.append((neighbor, (i, j), 'swap'))
        
        # INSERT operasyonu: bir haftalık bakımı başka yere kaydır
        for i in range(len(solution)):
            if solution[i] != 0:
                for j in range(max(0, i-5), min(len(solution), i+5)):
                    if i != j and solution[j] == 0:
                        if (i, j) not in self.tabu_list:
                            neighbor = solution.copy()
                            neighbor[j] = neighbor[i]
                            neighbor[i] = 0
                            neighbors.append((neighbor, (i, j), 'insert'))
        
        return neighbors  
        
    def update_tabu_list(self, move):
        """Tabu listesini güncelle"""
        self.tabu_list.append(move)
        
        # Tabu tenure'ı aşan hareketleri çıkar
        if len(self.tabu_list) > self.tabu_tenure:
            self.tabu_list.pop(0)  

    def run(self, initial_solution):
        """Tabu Search'ü çalıştır"""
        print("\n--- Tabu Search Başlatılıyor ---")
        start_time = datetime.now()
        
        current_solution = initial_solution.copy()
        current_fitness = self.calculate_fitness(current_solution)
        
        best_solution = current_solution.copy()
        best_fitness = current_fitness
        
        self.best_fitness_history.append(best_fitness)
        
        iterations_without_improvement = 0
        
        for iteration in range(self.max_iterations):
            # Komşuları üret
            neighbors = self.generate_neighbors(current_solution)
            
            if not neighbors:
                print(f"İterasyon {iteration}: Komşu bulunamadı, durduruluyor.")
                break
            
            # En iyi komşuyu bul
            best_neighbor = None
            best_neighbor_fitness = float('inf')
            best_move = None
            
            for neighbor, move, move_type in neighbors:
                neighbor_fitness = self.calculate_fitness(neighbor)
                
                # Aspiration kriteri: Tabu olsa bile şu ana kadarki en iyi çözümden iyiyse kabul et
                if neighbor_fitness < best_fitness:
                    best_neighbor = neighbor
                    best_neighbor_fitness = neighbor_fitness
                    best_move = move
                    break  # En iyi bulundu, döngüden çık
                
                # Normal komşu değerlendirmesi
                if neighbor_fitness < best_neighbor_fitness:
                    best_neighbor = neighbor
                    best_neighbor_fitness = neighbor_fitness
                    best_move = move
            
            # Komşuya geç
            current_solution = best_neighbor
            current_fitness = best_neighbor_fitness
            
            # Tabu listesini güncelle
            if best_move:
                self.update_tabu_list(best_move)
            
            # En iyi çözümü güncelle
            if current_fitness < best_fitness:
                best_solution = current_solution.copy()
                best_fitness = current_fitness
                iterations_without_improvement = 0
                print(f"TS İterasyon {iteration}: YENİ EN İYİ Fitness = {best_fitness:.2f} ✓")
            else:
                iterations_without_improvement += 1
            
            self.best_fitness_history.append(best_fitness)
            
            # Erken durdurma
            if iterations_without_improvement > 15:
                print(f"İterasyon {iteration}: 15 iterasyonda iyileşme yok, durduruluyor.")
                break
        
        end_time = datetime.now()
        computation_time = (end_time - start_time).total_seconds()
        
        print(f"\nTabu Search tamamlandı!")
        print(f"Süre: {computation_time:.2f} saniye")
        print(f"Başlangıç fitness: {self.calculate_fitness(initial_solution):.2f}")
        print(f"Final fitness: {best_fitness:.2f}")
        improvement = ((self.calculate_fitness(initial_solution) - best_fitness) / 
                       self.calculate_fitness(initial_solution) * 100)
        print(f"İyileşme: %{improvement:.2f}")
        
        return best_solution, best_fitness, computation_time        