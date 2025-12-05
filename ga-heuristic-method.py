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
        
# Visualizasyon fonksiyonları
def plot_fitness_evolution(ga):
    """Fitness evrimini görselleştir"""
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(ga.best_fitness_history, label='En İyi Fitness', linewidth=2)
    plt.plot(ga.avg_fitness_history, label='Ortalama Fitness', alpha=0.7)
    plt.xlabel('Nesil')
    plt.ylabel('Fitness Değeri')
    plt.title('GA Fitness Evrimi')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(ga.best_fitness_history, linewidth=2, color='green')
    plt.xlabel('Nesil')
    plt.ylabel('En İyi Fitness')
    plt.title('En İyi Çözümün Gelişimi')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def visualize_schedule(chromosome, problem):
    """Bakım çizelgesini görselleştir"""
    plt.figure(figsize=(15, 6))
    
    # Haftalık bakım durumunu göster
    schedule_matrix = np.zeros((problem.total_equipment, problem.T))
    
    for week in range(problem.T):
        if chromosome[week] != 0:
            eq_id = chromosome[week] - 1
            schedule_matrix[eq_id, week] = 1
    
    plt.imshow(schedule_matrix, aspect='auto', cmap='RdYlGn_r', interpolation='nearest')
    plt.colorbar(label='Bakım Durumu (1=Bakımda, 0=Çalışıyor)')
    plt.xlabel('Hafta')
    plt.ylabel('Ekipman ID')
    plt.title('Preventive Maintenance Schedule (GA)')
    plt.tight_layout()
    plt.show()


def analyze_production_demand_gap(chromosome, problem, ga):
    """Üretim-talep farkını analiz et"""
    _, water_gap, electricity_gap = ga.calculate_fitness(chromosome)
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # Su üretimi-talep analizi
    axes[0].plot(range(problem.T), water_gap, marker='o', linewidth=2, markersize=4)
    axes[0].axhline(y=0, color='r', linestyle='--', label='Hedef (Gap=0)')
    axes[0].fill_between(range(problem.T), 0, water_gap, alpha=0.3)
    axes[0].set_xlabel('Hafta')
    axes[0].set_ylabel('Su Üretim-Talep Farkı (MIGD)')
    axes[0].set_title('Su Üretim Fazlası/Açığı')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Elektrik üretimi-talep analizi
    axes[1].plot(range(problem.T), electricity_gap, marker='s', linewidth=2, 
                 markersize=4, color='orange')
    axes[1].axhline(y=0, color='r', linestyle='--', label='Hedef (Gap=0)')
    axes[1].fill_between(range(problem.T), 0, electricity_gap, alpha=0.3, color='orange')
    axes[1].set_xlabel('Hafta')
    axes[1].set_ylabel('Elektrik Üretim-Talep Farkı (MW)')
    axes[1].set_title('Elektrik Üretim Fazlası/Açığı')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # İstatistiksel analiz
    print("\n=== Üretim-Talep Farkı İstatistikleri ===")
    print(f"Su - Ortalama Gap: {np.mean(water_gap):.2f} MIGD")
    print(f"Su - Std Sapma: {np.std(water_gap):.2f} MIGD")
    print(f"Su - Min Gap: {np.min(water_gap):.2f} MIGD")
    print(f"Su - Max Gap: {np.max(water_gap):.2f} MIGD")
    print(f"\nElektrik - Ortalama Gap: {np.mean(electricity_gap):.2f} MW")
    print(f"Elektrik - Std Sapma: {np.std(electricity_gap):.2f} MW")
    print(f"Elektrik - Min Gap: {np.min(electricity_gap):.2f} MW")
    print(f"Elektrik - Max Gap: {np.max(electricity_gap):.2f} MW")             
        

# Ana çalıştırma
if __name__ == "__main__":
    # Problem oluştur
    problem = PMSProblem()
    
    # GA'yı çalıştır
    print("\nGenetik Algoritma başlatılıyor...")
    ga = GeneticAlgorithm(problem, pop_size=100, generations=200, crossover_rate=0.8, mutation_rate=0.2)
    best_solution, best_fitness, comp_time = ga.run()