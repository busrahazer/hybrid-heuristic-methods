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