import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from algorithms import GeneticAlgorithm

# Stil ayarları
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


# ==================== TEMEL GÖRSELLEŞTİRMELER ====================
class BasicVisualizations:
    """Temel GA vs GA+TS karşılaştırma grafikleri"""
    
    @staticmethod
    def plot_schedule(chromosome, problem, title="Bakım Çizelgesi"):
        """52 haftalık bakım çizelgesini görselleştir"""
        schedule_matrix = np.zeros((problem.total_equipment, problem.T))

        for week in range(problem.T):
            if chromosome[week] != 0:
                eq_id = chromosome[week] - 1
                schedule_matrix[eq_id, week] = 1

        plt.figure(figsize=(16, 8))
        plt.imshow(schedule_matrix, aspect='auto', cmap='RdYlGn_r', interpolation='nearest')
        plt.colorbar(label='Bakım Durumu (1=Bakımda, 0=Çalışıyor)')
        plt.xlabel('Hafta', fontsize=12, fontweight='bold')
        plt.ylabel('Ekipman ID', fontsize=12, fontweight='bold')
        plt.title(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_convergence_comparison(ga_obj, hybrid_obj):
        """GA vs GA+TS yakınsama karşılaştırması"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Sol: GA tek başına
        axes[0].plot(ga_obj.best_fitness_history, 'b-', linewidth=2, label='En İyi Fitness')
        axes[0].plot(ga_obj.avg_fitness_history, 'orange', linewidth=2, alpha=0.7, label='Ortalama Fitness')
        axes[0].set_xlabel('Nesil', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Fitness Değeri', fontsize=12, fontweight='bold')
        axes[0].set_title('GA (Tek Başına) Yakınsama', fontsize=13, fontweight='bold')
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3)

        # Sağ: GA+TS hibrit
        ga_gens = len(hybrid_obj.ga_fitness_history)
        ts_iters = len(hybrid_obj.ts_fitness_history)

        axes[1].plot(range(ga_gens), hybrid_obj.ga_fitness_history, 'b-', linewidth=2, label='GA Fazı')
        axes[1].plot(range(ga_gens, ga_gens + ts_iters), hybrid_obj.ts_fitness_history, 'g-', linewidth=2, label='TS Fazı')
        axes[1].axvline(x=ga_gens, color='red', linestyle='--', linewidth=2, alpha=0.7, label='GA → TS Geçişi')
        axes[1].set_xlabel('İterasyon / Nesil', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('En İyi Fitness', fontsize=12, fontweight='bold')
        axes[1].set_title('GA+TS (Hibrit) Yakınsama', fontsize=13, fontweight='bold')
        axes[1].legend(fontsize=11)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_gap_comparison(ga_solution, hybrid_solution, problem):
        """Su ve elektrik gap karşılaştırması"""
        ga_obj = GeneticAlgorithm(problem, verbose=False)
        _, ga_details = ga_obj.calculate_fitness(ga_solution)
        _, hybrid_details = ga_obj.calculate_fitness(hybrid_solution)

        fig, axes = plt.subplots(2, 1, figsize=(16, 10))

        weeks = range(52)

        # Su gap
        axes[0].plot(weeks, ga_details['water_gap'], 'o-', label='GA Gap', 
                     color='steelblue', linewidth=2, markersize=5, alpha=0.7)
        axes[0].plot(weeks, hybrid_details['water_gap'], 's-', label='GA+TS Gap', 
                     color='darkgreen', linewidth=2, markersize=5, alpha=0.7)
        axes[0].axhline(y=0, color='red', linestyle='--', linewidth=2, label='Kritik Eşik (Gap=0)')
        axes[0].fill_between(weeks, 0, ga_details['water_gap'], alpha=0.2, color='steelblue')
        axes[0].fill_between(weeks, 0, hybrid_details['water_gap'], alpha=0.2, color='darkgreen')
        axes[0].axvspan(20, 32, alpha=0.1, color='orange', label='Yaz Ayları')
        axes[0].set_xlabel('Hafta', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Su Üretim-Talep Farkı (MIGD)', fontsize=12, fontweight='bold')
        axes[0].set_title('Haftalık Su Üretim-Talep Dengesi', fontsize=13, fontweight='bold')
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3)

        # Elektrik gap
        axes[1].plot(weeks, ga_details['electricity_gap'], 'o-', label='GA Gap', 
                     color='steelblue', linewidth=2, markersize=5, alpha=0.7)
        axes[1].plot(weeks, hybrid_details['electricity_gap'], 's-', label='GA+TS Gap', 
                     color='darkgreen', linewidth=2, markersize=5, alpha=0.7)
        axes[1].axhline(y=0, color='red', linestyle='--', linewidth=2, label='Kritik Eşik (Gap=0)')
        axes[1].fill_between(weeks, 0, ga_details['electricity_gap'], alpha=0.2, color='steelblue')
        axes[1].fill_between(weeks, 0, hybrid_details['electricity_gap'], alpha=0.2, color='darkgreen')
        axes[1].axvspan(20, 32, alpha=0.1, color='orange')
        axes[1].set_xlabel('Hafta', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Elektrik Üretim-Talep Farkı (MW)', fontsize=12, fontweight='bold')
        axes[1].set_title('Haftalık Elektrik Üretim-Talep Dengesi', fontsize=13, fontweight='bold')
        axes[1].legend(fontsize=11)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # İstatistikler
        print("\n" + "="*70)
        print("MİN GAP İSTATİSTİKLERİ")
        print("="*70)
        print(f"Su Min Gap:")
        print(f"  GA: {ga_details['min_water_gap']:.2f} MIGD")
        print(f"  GA+TS: {hybrid_details['min_water_gap']:.2f} MIGD")
        improvement_water = ((hybrid_details['min_water_gap'] - ga_details['min_water_gap']) / abs(ga_details['min_water_gap']) * 100)
        print(f"  İyileşme: {improvement_water:+.2f}%")
        print(f"\nElektrik Min Gap:")
        print(f"  GA: {ga_details['min_electricity_gap']:.2f} MW")
        print(f"  GA+TS: {hybrid_details['min_electricity_gap']:.2f} MW")
        improvement_elec = ((hybrid_details['min_electricity_gap'] - ga_details['min_electricity_gap']) / abs(ga_details['min_electricity_gap']) * 100)
        print(f"  İyileşme: {improvement_elec:+.2f}%")
        print("="*70)
    