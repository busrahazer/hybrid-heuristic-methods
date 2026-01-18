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
    
    @staticmethod
    def plot_comparison_table(ga_results, hybrid_results):
        """Karşılaştırma tablosu"""
        ga_solution, ga_fitness, ga_time = ga_results
        hybrid_solution, hybrid_fitness, hybrid_time, _, _, _ = hybrid_results

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

        return df
    
# ==================== GELİŞMİŞ GÖRSELLEŞTİRMELER ====================
class AdvancedVisualizations:
    """Gelişmiş analiz grafikleri"""
    
    @staticmethod
    def plot_diversity_evolution(ga_obj):
        """Popülasyon çeşitliliğinin evrimiGA nesiller boyunca"""
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        # Çeşitlilik evrimi
        axes[0].plot(ga_obj.diversity_history, 'purple', linewidth=2, label='Popülasyon Çeşitliliği')
        axes[0].set_xlabel('Nesil', fontsize=11, fontweight='bold')
        axes[0].set_ylabel('Çeşitlilik (Hamming Distance)', fontsize=11, fontweight='bold')
        axes[0].set_title('Popülasyon Çeşitliliğinin Evrimi', fontsize=12, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)

        # Her nesildeki iyileştirme
        axes[1].bar(range(len(ga_obj.improvement_per_generation)), ga_obj.improvement_per_generation, 
                    color='green', alpha=0.6, label='İyileşme Miktarı')
        axes[1].set_xlabel('Nesil', fontsize=11, fontweight='bold')
        axes[1].set_ylabel('İyileşme (Fitness Azalması)', fontsize=11, fontweight='bold')
        axes[1].set_title('Nesil Bazlı İyileşme Katkıları', fontsize=12, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_optimization_progress(optimizer):
        """Optuna optimizasyon ilerlemesi"""
        df = optimizer.get_optimization_dataframe()

        fig, axes = plt.subplots(2, 2, figsize=(16, 10))

        # 1. Trial bazlı fitness
        axes[0, 0].plot(df['trial'], df['fitness'], 'o-', linewidth=2, markersize=6, alpha=0.7, label='Fitness')
        axes[0, 0].plot(df['trial'], df['fitness'].cummin(), 'r-', linewidth=2, label='En İyi (Kümülatif)')
        axes[0, 0].set_xlabel('Trial Numarası', fontsize=11, fontweight='bold')
        axes[0, 0].set_ylabel('Fitness Değeri', fontsize=11, fontweight='bold')
        axes[0, 0].set_title('Optimizasyon İlerlemesi', fontsize=12, fontweight='bold')
        axes[0, 0].legend(fontsize=10)
        axes[0, 0].grid(True, alpha=0.3)

        # 2. GA vs Hibrit iyileşme
        axes[0, 1].scatter(df['ga_fitness'], df['fitness'], alpha=0.6, s=50)
        axes[0, 1].plot([df['ga_fitness'].min(), df['ga_fitness'].max()], 
                        [df['ga_fitness'].min(), df['ga_fitness'].max()], 
                        'r--', linewidth=2, label='Eşitlik Çizgisi')
        axes[0, 1].set_xlabel('GA Fitness', fontsize=11, fontweight='bold')
        axes[0, 1].set_ylabel('Hibrit Fitness', fontsize=11, fontweight='bold')
        axes[0, 1].set_title('GA vs Hibrit Performans', fontsize=12, fontweight='bold')
        axes[0, 1].legend(fontsize=10)
        axes[0, 1].grid(True, alpha=0.3)

        # 3. İyileşme dağılımı
        axes[1, 0].hist(df['improvement'], bins=20, color='green', alpha=0.7, edgecolor='black')
        axes[1, 0].axvline(df['improvement'].mean(), color='red', linestyle='--', linewidth=2, label=f'Ortalama: {df["improvement"].mean():.2f}')
        axes[1, 0].set_xlabel('İyileşme Miktarı (GA → Hibrit)', fontsize=11, fontweight='bold')
        axes[1, 0].set_ylabel('Frekans', fontsize=11, fontweight='bold')
        axes[1, 0].set_title('TS İyileşme Dağılımı', fontsize=12, fontweight='bold')
        axes[1, 0].legend(fontsize=10)
        axes[1, 0].grid(axis='y', alpha=0.3)

        # 4. Hesaplama süresi dağılımı
        axes[1, 1].boxplot([df['time']], labels=['Hibrit'])
        axes[1, 1].set_ylabel('Süre (saniye)', fontsize=11, fontweight='bold')
        axes[1, 1].set_title(f'Hesaplama Süresi Dağılımı\n(Ort: {df["time"].mean():.2f}s)', fontsize=12, fontweight='bold')
        axes[1, 1].grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_parameter_importance(optimizer):
        """Parametre önem grafiği"""
        importance = optimizer.get_parameter_importance()

        fig, ax = plt.subplots(figsize=(12, 6))

        params = list(importance.keys())
        scores = list(importance.values())

        colors = ['steelblue' if 'ga' in p or p in ['pop_size', 'generations', 'crossover_rate', 'mutation_rate'] 
                  else 'darkgreen' for p in params]

        bars = ax.barh(params, scores, color=colors, alpha=0.7, edgecolor='black')

        ax.set_xlabel('Önem Skoru', fontsize=12, fontweight='bold')
        ax.set_title('Parametre Önem Sıralaması (Optuna)', fontsize=13, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)

        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='steelblue', label='GA Parametreleri'),
            Patch(facecolor='darkgreen', label='TS Parametreleri')
        ]
        ax.legend(handles=legend_elements, fontsize=10)

        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_sensitivity_analysis(sensitivity_results):
        """Sensitivity analiz grafikleri"""
        n_params = len(sensitivity_results)
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        axes = axes.flatten()

        for idx, (param_name, results) in enumerate(sensitivity_results.items()):
            if idx >= 4:
                break

            values = [r['value'] for r in results]
            avg_fitness = [r['avg_fitness'] for r in results]
            std_fitness = [r['std_fitness'] for r in results]

            axes[idx].errorbar(values, avg_fitness, yerr=std_fitness, 
                               marker='o', linewidth=2, markersize=8, capsize=5)
            axes[idx].set_xlabel(param_name, fontsize=11, fontweight='bold')
            axes[idx].set_ylabel('Fitness Değeri', fontsize=11, fontweight='bold')
            axes[idx].set_title(f'{param_name} Sensitivity', fontsize=12, fontweight='bold')
            axes[idx].grid(True, alpha=0.3)

            # En iyi değeri işaretle
            best_idx = np.argmin(avg_fitness)
            axes[idx].scatter([values[best_idx]], [avg_fitness[best_idx]], 
                              color='red', s=200, marker='*', zorder=5, label='En İyi')
            axes[idx].legend(fontsize=9)

        plt.tight_layout()
        plt.show()