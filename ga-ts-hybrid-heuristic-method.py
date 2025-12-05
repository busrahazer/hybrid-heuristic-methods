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
    
class HybridGATS:
    """Hibrit GA + TS Yaklaşımı"""
    
    def __init__(self, problem, ga_params=None, ts_params=None):
        self.problem = problem
        self.ga_params = ga_params or {}
        self.ts_params = ts_params or {}
        
        self.ga_fitness_history = []
        self.ts_fitness_history = []
        self.ga_time = 0
        self.ts_time = 0
        
    def run(self):
        """Hibrit yaklaşımı çalıştır"""
        
        # 1. Adım: GA ile başlangıç çözümü bul
        ga = GeneticAlgorithm(self.problem, **self.ga_params)
        ga_solution, ga_fitness, ga_time = ga.run()
        
        self.ga_fitness_history = ga.best_fitness_history
        self.ga_time = ga_time
        
        print(f"\nGA Tamamlandı: Fitness = {ga_fitness:.2f}, Süre = {ga_time:.2f}s")
        
        # 2. Adım: TS ile iyileştir
        ts = TabuSearch(self.problem, **self.ts_params)
        final_solution, final_fitness, ts_time = ts.run(ga_solution)
        
        self.ts_fitness_history = ts.best_fitness_history
        self.ts_time = ts_time
        
        print(f"\nTS Tamamlandı: Fitness = {final_fitness:.2f}, Süre = {ts_time:.2f}s")
        
        # Toplam iyileşme
        total_improvement = ((ga_fitness - final_fitness) / ga_fitness * 100)
        total_time = ga_time + ts_time
        
        return final_solution, final_fitness, total_time, ga_solution, ga_fitness 

# Karşılaştırma ve Görselleştirme
def compare_ga_vs_hybrid(ga_only_results, hybrid_results, problem):
    """GA-only ve GA+TS hibrit sonuçlarını karşılaştır"""
    
    ga_solution, ga_fitness, ga_time = ga_only_results
    hybrid_solution, hybrid_fitness, hybrid_time, _, _ = hybrid_results
    
    # Karşılaştırma tablosu
    comparison_data = {
        'Metrik': ['Fitness Değeri', 'Hesaplama Süresi (s)', 'İyileşme Oranı (%)'],
        'GA (Tek Başına)': [
            f"{ga_fitness:.2f}",
            f"{ga_time:.2f}",
            "-"
        ],
        'GA + TS (Hibrit)': [
            f"{hybrid_fitness:.2f}",
            f"{hybrid_time:.2f}",
            f"{((ga_fitness - hybrid_fitness) / ga_fitness * 100):.2f}%"
        ]
    }
    
    df = pd.DataFrame(comparison_data)
    print("\n" + "="*70)
    print("KARŞILAŞTIRMA TABLOSU: GA vs GA+TS")
    print("="*70)
    print(df.to_string(index=False))
    print("="*70)
    
    # Grafik karşılaştırma
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # 1. Fitness karşılaştırması (bar chart)
    methods = ['GA\n(Tek Başına)', 'GA + TS\n(Hibrit)']
    fitness_values = [ga_fitness, hybrid_fitness]
    colors = ['steelblue', 'darkgreen']
    
    axes[0, 0].bar(methods, fitness_values, color=colors, alpha=0.7, edgecolor='black')
    axes[0, 0].set_ylabel('Fitness Değeri (Düşük = İyi)', fontsize=11)
    axes[0, 0].set_title('Fitness Değeri Karşılaştırması', fontsize=12, fontweight='bold')
    axes[0, 0].grid(axis='y', alpha=0.3)
    for i, v in enumerate(fitness_values):
        axes[0, 0].text(i, v + max(fitness_values)*0.02, f'{v:.2f}', 
                        ha='center', fontweight='bold')
    
    # 2. Hesaplama süresi karşılaştırması
    time_values = [ga_time, hybrid_time]
    axes[0, 1].bar(methods, time_values, color=colors, alpha=0.7, edgecolor='black')
    axes[0, 1].set_ylabel('Süre (saniye)', fontsize=11)
    axes[0, 1].set_title('Hesaplama Süresi Karşılaştırması', fontsize=12, fontweight='bold')
    axes[0, 1].grid(axis='y', alpha=0.3)
    for i, v in enumerate(time_values):
        axes[0, 1].text(i, v + max(time_values)*0.02, f'{v:.2f}s', 
                        ha='center', fontweight='bold')
    
    # 3. İyileşme yüzdesi
    improvement = ((ga_fitness - hybrid_fitness) / ga_fitness * 100)
    axes[1, 0].bar(['İyileşme'], [improvement], color='green', alpha=0.7, edgecolor='black')
    axes[1, 0].set_ylabel('İyileşme Yüzdesi (%)', fontsize=11)
    axes[1, 0].set_title('TS ile Sağlanan İyileşme', fontsize=12, fontweight='bold')
    axes[1, 0].grid(axis='y', alpha=0.3)
    axes[1, 0].text(0, improvement + 0.5, f'%{improvement:.2f}', 
                    ha='center', fontweight='bold', fontsize=14)
    
    # 4. Üretim-talep gap karşılaştırması
    ga_obj = GeneticAlgorithm(problem)
    _, ga_water_gap, ga_elec_gap = ga_obj.calculate_fitness(ga_solution)
    _, hybrid_water_gap, hybrid_elec_gap = ga_obj.calculate_fitness(hybrid_solution)
    
    gap_comparison = {
        'GA - Su Std': np.std(ga_water_gap),
        'Hibrit - Su Std': np.std(hybrid_water_gap),
        'GA - Elektrik Std': np.std(ga_elec_gap),
        'Hibrit - Elektrik Std': np.std(hybrid_elec_gap)
    }
    
    labels = list(gap_comparison.keys())
    values = list(gap_comparison.values())
    bar_colors = ['steelblue', 'darkgreen', 'steelblue', 'darkgreen']
    
    axes[1, 1].bar(range(len(labels)), values, color=bar_colors, alpha=0.7, edgecolor='black')
    axes[1, 1].set_xticks(range(len(labels)))
    axes[1, 1].set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    axes[1, 1].set_ylabel('Standart Sapma', fontsize=11)
    axes[1, 1].set_title('Üretim-Talep Gap Standart Sapması', fontsize=12, fontweight='bold')
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return df


def plot_hybrid_convergence(hybrid_obj):
    """Hibrit yaklaşımın yakınsama grafiği"""
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    
    # GA + TS birleşik grafik
    ga_gens = len(hybrid_obj.ga_fitness_history)
    ts_iters = len(hybrid_obj.ts_fitness_history)
    
    all_fitness = hybrid_obj.ga_fitness_history + hybrid_obj.ts_fitness_history
    
    axes[0].plot(range(ga_gens), hybrid_obj.ga_fitness_history, 
                 'b-', linewidth=2, label='GA Fazı')
    axes[0].plot(range(ga_gens, ga_gens + ts_iters), hybrid_obj.ts_fitness_history, 
                 'g-', linewidth=2, label='TS Fazı')
    axes[0].axvline(x=ga_gens, color='red', linestyle='--', 
                    label='GA → TS Geçişi', linewidth=2)
    axes[0].set_xlabel('İterasyon / Nesil', fontsize=11)
    axes[0].set_ylabel('En İyi Fitness', fontsize=11)
    axes[0].set_title('Hibrit GA+TS Yakınsama Grafiği', fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Faz karşılaştırması
    phases = ['GA Fazı', 'TS Fazı']
    phase_improvements = [
        hybrid_obj.ga_fitness_history[0] - hybrid_obj.ga_fitness_history[-1],
        hybrid_obj.ts_fitness_history[0] - hybrid_obj.ts_fitness_history[-1]
    ]
    
    axes[1].bar(phases, phase_improvements, color=['steelblue', 'darkgreen'], 
                alpha=0.7, edgecolor='black')
    axes[1].set_ylabel('Fitness İyileşme Miktarı', fontsize=11)
    axes[1].set_title('Faz Bazlı İyileşme Katkıları', fontsize=12, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)
    
    for i, v in enumerate(phase_improvements):
        axes[1].text(i, v + max(phase_improvements)*0.02, f'{v:.2f}', 
                     ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.show()      

if __name__ == "__main__":
    # Problem oluştur
    problem = PMSProblem()  

    # Sadece GA
    print("\nSadece Genetik Algoritma (GA) çalıştırılıyor...")
    ga_only = GeneticAlgorithm(problem, pop_size=100, generations=200)
    ga_solution, ga_fitness, ga_time = ga_only.run()
    ga_only_results = (ga_solution, ga_fitness, ga_time)
    
    # GA + TS Hibrit
    print("\nGA + TS Hibrit yaklaşım çalıştırılıyor...")
    hybrid = HybridGATS(
        problem, 
        ga_params={'pop_size': 100, 'generations': 200},
        ts_params={'tabu_tenure': 4, 'max_iterations': 50}
    )
    hybrid_solution, hybrid_fitness, hybrid_time, _, _ = hybrid.run()
    hybrid_results = (hybrid_solution, hybrid_fitness, hybrid_time, ga_solution, ga_fitness)

    # KARŞILAŞTIRMA
    print("\n" + "="*70)
    print("SONUÇLARI KARŞILAŞTIR VE GÖRSELLEŞTİR")
    print("="*70)
    
    comparison_df = compare_ga_vs_hybrid(ga_only_results, hybrid_results, problem)
    plot_hybrid_convergence(hybrid)
    
    print("\nGA + TS TAMAMLANDI!")
    print(f"GA Fitness: {ga_fitness:.2f}")
    print(f"Hibrit (GA+TS) Fitness: {hybrid_fitness:.2f}")
    print(f"İyileşme: %{((ga_fitness - hybrid_fitness) / ga_fitness * 100):.2f}")  
    
       