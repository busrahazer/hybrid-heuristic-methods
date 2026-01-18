import numpy as np
import pandas as pd
from algorithms import PMSProblem, GeneticAlgorithm, HybridGATS
from optimization import ParameterOptimizer, SensitivityAnalyzer
from visualization import BasicVisualizations, AdvancedVisualizations, ReportGenerator

# Global ayarlar
GLOBAL_SEED = 49
np.random.seed(GLOBAL_SEED)


def run_basic_comparison(problem):
    """BÖLÜM 1: Temel GA vs GA+TS Karşılaştırması"""
    print("\n" + "="*80)
    print("BÖLÜM 1: TEMEL KARŞILAŞTIRMA (GA vs GA+TS)")
    print("="*80)

    # Standart parametreler
    ga_params = {'pop_size': 100, 'generations': 200}
    ts_params = {'tabu_tenure': 4, 'max_iterations': 50}

    # Sadece GA
    print("\n[1/2] Sadece GA çalıştırılıyor...")
    ga_only = GeneticAlgorithm(problem, **ga_params, verbose=True)
    ga_solution, ga_fitness, ga_time, ga_details = ga_only.run()
    ga_results = (ga_solution, ga_fitness, ga_time)

    # GA + TS Hibrit
    print("\n[2/2] GA + TS Hibrit çalıştırılıyor...")
    hybrid = HybridGATS(problem, ga_params=ga_params, ts_params=ts_params, verbose=True)
    hybrid_solution, hybrid_fitness, hybrid_time, _, _, _ = hybrid.run()
    hybrid_results = (hybrid_solution, hybrid_fitness, hybrid_time, ga_solution, ga_fitness, ga_details)

    # Görselleştirmeler
    print("\n Temel Görselleştirmeler Oluşturuluyor...")
    
    # 1. Karşılaştırma tablosu
    BasicVisualizations.plot_comparison_table(ga_results, hybrid_results)
    
    # 2. Yakınsama grafikleri
    print("\n1 Yakınsama Grafikleri:")
    BasicVisualizations.plot_convergence_comparison(ga_only, hybrid)
    
    # 3. Bakım çizelgeleri
    print("\n 2Bakım Çizelgeleri:")
    BasicVisualizations.plot_schedule(ga_solution, problem, title="Bakım Çizelgesi (GA)")
    BasicVisualizations.plot_schedule(hybrid_solution, problem, title="Bakım Çizelgesi (GA+TS)")
    
    # 4. Gap analizi
    print("\n3 Üretim-Talep Gap Analizi:")
    BasicVisualizations.plot_gap_comparison(ga_solution, hybrid_solution, problem)
    
    # 5. Çeşitlilik evrimi
    print("\n4 Popülasyon Çeşitliliği ve İyileşme:")
    AdvancedVisualizations.plot_diversity_evolution(ga_only)

    return ga_results, hybrid_results
