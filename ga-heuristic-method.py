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
        base_demand = np.pnes(self.T)
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
        
             