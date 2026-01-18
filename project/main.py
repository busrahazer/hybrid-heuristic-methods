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

def run_parameter_optimization(problem, n_trials=30):
    """BÖLÜM 2: Parametre Optimizasyonu"""
    print("\n" + "="*80)
    print("BÖLÜM 2: PARAMETRE OPTİMİZASYONU (OPTUNA)")
    print("="*80)

    # Optuna optimizasyonu
    optimizer = ParameterOptimizer(problem, n_trials=n_trials, seed=GLOBAL_SEED)
    best_params = optimizer.run_optimization()

    # Parametre önemi
    print("\n Parametre Önem Analizi:")
    importance = optimizer.get_parameter_importance()

    # Yakınsama analizi
    convergence_info = optimizer.analyze_convergence()

    # Görselleştirmeler
    print("\n Optimizasyon Görselleştirmeleri:")
    
    print("\n1 Optimizasyon İlerlemesi:")
    AdvancedVisualizations.plot_optimization_progress(optimizer)
    
    print("\n2 Parametre Önem Sıralaması:")
    AdvancedVisualizations.plot_parameter_importance(optimizer)

    optimization_results = {
        'best_params': best_params,
        'best_value': optimizer.study.best_value,
        'n_trials': n_trials,
        'convergence_info': convergence_info
    }

    return best_params, optimization_results


def run_optimized_comparison(problem, best_params):
    """BÖLÜM 3: Optimize Edilmiş Parametrelerle Final Karşılaştırma"""
    print("\n" + "="*80)
    print("BÖLÜM 3: OPTİMİZE EDİLMİŞ PARAMETRELERLE FİNAL KARŞILAŞTIRMA")
    print("="*80)
    
    #  ADİL KARŞILAŞTIRMA: Aynı seed ile yeni problem oluştur
    comparison_seed = GLOBAL_SEED + 999  # Farklı ama sabit bir seed
    print(f"\n Karşılaştırma Seed'i: {comparison_seed} (Standart ve Optimize için AYNI)")

    # GA parametrelerini ayır
    ga_params_opt = {
        'pop_size': best_params['pop_size'],
        'generations': best_params['generations'],
        'crossover_rate': best_params['crossover_rate'],
        'mutation_rate': best_params['mutation_rate']
    }

    # TS parametrelerini ayır
    ts_params_opt = {
        'tabu_tenure': best_params['tabu_tenure'],
        'max_iterations': best_params['max_iterations']
    }
    
    # Standart parametreler
    ga_params_std = {'pop_size': 100, 'generations': 200, 'crossover_rate': 0.8, 'mutation_rate': 0.2}
    ts_params_std = {'tabu_tenure': 4, 'max_iterations': 50}

    print(f"\n Standart Parametreler:")
    print(f"   GA: {ga_params_std}")
    print(f"   TS: {ts_params_std}")
    print(f"\n Optimize Parametreler:")
    print(f"   GA: {ga_params_opt}")
    print(f"   TS: {ts_params_opt}")

    #  STANDART PARAMETRELERLE TEST
    print("\n[1/4] Standart GA çalıştırılıyor...")
    np.random.seed(comparison_seed)
    problem_std = PMSProblem(seed=comparison_seed)
    ga_std = GeneticAlgorithm(problem_std, **ga_params_std, verbose=False)
    ga_std_solution, ga_std_fitness, ga_std_time, _ = ga_std.run()

    print("[2/4] Standart Hibrit çalıştırılıyor...")
    np.random.seed(comparison_seed)
    problem_std2 = PMSProblem(seed=comparison_seed)
    hybrid_std = HybridGATS(problem_std2, ga_params=ga_params_std, ts_params=ts_params_std, verbose=False)
    hybrid_std_solution, hybrid_std_fitness, hybrid_std_time, _, _, _ = hybrid_std.run()

    #  OPTİMİZE PARAMETRELERLE TEST
    print("[3/4] Optimize GA çalıştırılıyor...")
    np.random.seed(comparison_seed)
    problem_opt = PMSProblem(seed=comparison_seed)
    ga_opt = GeneticAlgorithm(problem_opt, **ga_params_opt, verbose=False)
    ga_opt_solution, ga_opt_fitness, ga_opt_time, ga_opt_details = ga_opt.run()
    ga_opt_results = (ga_opt_solution, ga_opt_fitness, ga_opt_time)

    print("[4/4] Optimize Hibrit çalıştırılıyor...")
    np.random.seed(comparison_seed)
    problem_opt2 = PMSProblem(seed=comparison_seed)
    hybrid_opt = HybridGATS(problem_opt2, ga_params=ga_params_opt, ts_params=ts_params_opt, verbose=False)
    hybrid_opt_solution, hybrid_opt_fitness, hybrid_opt_time, _, _, _ = hybrid_opt.run()
    hybrid_opt_results = (hybrid_opt_solution, hybrid_opt_fitness, hybrid_opt_time, 
                          ga_opt_solution, ga_opt_fitness, ga_opt_details)

    # KARŞILAŞTIRMA TABLOSU
    print("\n" + "="*80)
    print("STANDART vs OPTİMİZE PARAMETRELERİ DETAYLI KARŞILAŞTIRMA")
    print("="*80)
    
    comparison_df = pd.DataFrame({
        'Metrik': ['GA Fitness', 'Hibrit Fitness', 'GA Süresi (s)', 'Hibrit Süresi (s)'],
        'Standart': [
            f"{ga_std_fitness:.2f}",
            f"{hybrid_std_fitness:.2f}",
            f"{ga_std_time:.2f}",
            f"{hybrid_std_time:.2f}"
        ],
        'Optimize': [
            f"{ga_opt_fitness:.2f}",
            f"{hybrid_opt_fitness:.2f}",
            f"{ga_opt_time:.2f}",
            f"{hybrid_opt_time:.2f}"
        ],
        'İyileşme': [
            f"{((ga_std_fitness - ga_opt_fitness) / ga_std_fitness * 100):.2f}%",
            f"{((hybrid_std_fitness - hybrid_opt_fitness) / hybrid_std_fitness * 100):.2f}%",
            f"{((ga_std_time - ga_opt_time) / ga_std_time * 100):.2f}%",
            f"{((hybrid_std_time - hybrid_opt_time) / hybrid_std_time * 100):.2f}%"
        ]
    })
    
    print(comparison_df.to_string(index=False))
    print("="*80)
    
    # Görselleştirmeler
    print("\n Final Görselleştirmeleri:")
    BasicVisualizations.plot_comparison_table(ga_opt_results, hybrid_opt_results)
    BasicVisualizations.plot_convergence_comparison(ga_opt, hybrid_opt)
    BasicVisualizations.plot_gap_comparison(ga_opt_solution, hybrid_opt_solution, problem_opt2)

    return ga_opt_results, hybrid_opt_results, comparison_df


def run_sensitivity_analysis(problem, best_params):
    """BÖLÜM 4: Sensitivity Analizi (Opsiyonel - Uzun Sürebilir)"""
    print("\n" + "="*80)
    print("BÖLÜM 4: SENSİTİVİTE ANALİZİ")
    print("="*80)
    print("  UYARI: Bu analiz uzun sürebilir (10-20 dakika)")
    
    response = input("Devam etmek istiyor musunuz? (e/h): ")
    
    if response.lower() != 'e':
        print("Sensitivity analizi atlandı.")
        return None

    # Temel parametreler
    base_params = {
        'ga_params': {
            'pop_size': best_params['pop_size'],
            'generations': best_params['generations'],
            'crossover_rate': best_params['crossover_rate'],
            'mutation_rate': best_params['mutation_rate']
        },
        'ts_params': {
            'tabu_tenure': best_params['tabu_tenure'],
            'max_iterations': best_params['max_iterations']
        }
    }

    analyzer = SensitivityAnalyzer(problem, base_params)
    sensitivity_results = analyzer.run_full_analysis()
    
    best_values = analyzer.get_best_values()

    # Görselleştirme
    print("\n Sensitivity Grafikleri:")
    AdvancedVisualizations.plot_sensitivity_analysis(sensitivity_results)

    return sensitivity_results

def main():
    """Ana program"""
    print("\n" + "="*80)
    print(" " * 20 + "GA + TS + OPTUNA TAM ENTEGRE SİSTEM")
    print(" " * 15 + "Preventive Maintenance Scheduling (PMS)")
    print("="*80)

    # Problem oluştur
    problem = PMSProblem(seed=GLOBAL_SEED)
    print(f"\n Problem Parametreleri:")
    print(f"   • Toplam Ekipman: {problem.total_equipment}")
    print(f"   • Zaman Horizonu: {problem.T} hafta")
    print(f"   • Seed: {GLOBAL_SEED} (Tekrarlanabilir sonuçlar)")

    # ==================== BÖLÜM 1: Temel Karşılaştırma ====================
    ga_results, hybrid_results = run_basic_comparison(problem)

    # ==================== BÖLÜM 2: Parametre Optimizasyonu ====================
    print("\n" + "="*80)
    response = input("\nParametre optimizasyonu yapmak istiyor musunuz? (e/h): ")
    
    if response.lower() == 'e':
        n_trials = int(input("Kaç trial yapmak istersiniz? (önerilen: 30-50): "))
        best_params, optimization_results = run_parameter_optimization(problem, n_trials)

        # ==================== BÖLÜM 3: Optimize Parametrelerle Final ====================
        response2 = input("\nOptimize parametrelerle final karşılaştırma yapmak istiyor musunuz? (e/h): ")
        if response2.lower() == 'e':
            ga_opt_results, hybrid_opt_results, comparison_df = run_optimized_comparison(problem, best_params)

            # Standart vs Optimize karşılaştırması
            print("\n" + "="*80)
            print("STANDART vs OPTİMİZE PARAMETRELERİ KARŞILAŞTIRMA")
            print("="*80)
            print(f"\nStandart Hibrit Fitness: {hybrid_results[1]:.2f}")
            print(f"Optimize Hibrit Fitness: {hybrid_opt_results[1]:.2f}")
            improvement = ((hybrid_results[1] - hybrid_opt_results[1]) / hybrid_results[1] * 100)
            print(f"İyileşme: {improvement:.2f}%")
            print("="*80)
        else:
            optimization_results = None
    else:
        optimization_results = None

    # ==================== ÖZET RAPOR ====================
    print("\n" + "="*80)
    print(" " * 30 + "ÖZET RAPOR")
    print("="*80)
    ReportGenerator.generate_summary_report(ga_results, hybrid_results, optimization_results)

    print("\n TÜM ANALIZLER TAMAMLANDI!")
    print("="*80)


if __name__ == "__main__":
    main()