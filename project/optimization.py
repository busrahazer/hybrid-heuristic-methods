import optuna
import numpy as np
import pandas as pd
from algorithms import HybridGATS


# ==================== OPTUNA OPTİMİZASYON ====================
class ParameterOptimizer:
    """Optuna ile parametre optimizasyonu"""
    
    def __init__(self, problem, n_trials=30, seed=42):
        self.problem = problem
        self.n_trials = n_trials
        self.seed = seed
        
        self.study = None
        self.best_params = None
        self.optimization_history = []
        
    def objective(self, trial):
        """Optuna objective fonksiyonu"""
        
        #  HER TRIAL için FARKLI random state (ama AYNI problem)
        trial_seed = self.seed + trial.number * 1000  # Büyük aralıklarla seed değiştir
        np.random.seed(trial_seed)
        
        # GA parametreleri
        ga_params = {
            "pop_size": trial.suggest_int("pop_size", 50, 150),
            "generations": trial.suggest_int("generations", 100, 300),
            "crossover_rate": trial.suggest_float("crossover_rate", 0.6, 0.95),
            "mutation_rate": trial.suggest_float("mutation_rate", 0.05, 0.3),
        }

        # TS parametreleri
        ts_params = {
            "tabu_tenure": trial.suggest_int("tabu_tenure", 2, 10),
            "max_iterations": trial.suggest_int("max_iterations", 30, 80),
        }

        #  Hibrit algoritmayı çalıştır (aynı problem, farklı random başlangıç)
        hybrid = HybridGATS(self.problem, ga_params=ga_params, ts_params=ts_params, verbose=False)
        final_solution, final_fitness, total_time, ga_solution, ga_fitness, _ = hybrid.run()
        
        # İstatistikleri kaydet
        self.optimization_history.append({
            'trial': trial.number,
            'fitness': final_fitness,
            'ga_fitness': ga_fitness,
            'improvement': ga_fitness - final_fitness,
            'time': total_time,
            'trial_seed': trial_seed,  #  Seed'i kaydet
            **ga_params,
            **ts_params
        })

        return final_fitness
    
    def run_optimization(self):
        """Optimizasyonu çalıştır"""
        print("\n" + "="*80)
        print("OPTUNA İLE PARAMETRE OPTİMİZASYONU")
        print("="*80)
        print(f"Trial sayısı: {self.n_trials}")
        print(f"Seed: {self.seed}")
        print("Lütfen bekleyin, bu işlem birkaç dakika sürebilir...")
        print("="*80)
        
        # Optuna study oluştur
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        self.study = optuna.create_study(
            direction="minimize",
            study_name="PMS_Optimization",
            sampler=optuna.samplers.TPESampler(seed=self.seed)
        )
        
        # Optimizasyonu çalıştır
        self.study.optimize(self.objective, n_trials=self.n_trials, show_progress_bar=True)
        
        self.best_params = self.study.best_params
        
        # Sonuçları yazdır
        print("\n" + "="*80)
        print("OPTİMİZASYON TAMAMLANDI!")
        print("="*80)
        print(f"\n EN İYİ FITNESS: {self.study.best_value:.2f}")
        print("\n EN İYİ PARAMETRELER:")
        print("-" * 40)
        for k, v in self.best_params.items():
            print(f"  {k:20s}: {v}")
        print("="*80)
        
        return self.best_params
    
    def get_optimization_dataframe(self):
        """Optimizasyon geçmişini DataFrame olarak döndür"""
        return pd.DataFrame(self.optimization_history)
    
    def get_parameter_importance(self):
        """Parametre önem sıralaması"""
        if self.study is None:
            raise ValueError("Önce optimizasyonu çalıştırın!")
        
        importance = optuna.importance.get_param_importances(self.study)
        
        print("\n" + "="*80)
        print("PARAMETRE ÖNEM SIRALALAMASI")
        print("="*80)
        for param, score in importance.items():
            print(f"  {param:20s}: {score:.4f}")
        print("="*80)
        
        return importance
    
    def analyze_convergence(self):
        """Optimizasyon yakınsamasını analiz et"""
        df = self.get_optimization_dataframe()
        
        print("\n" + "="*80)
        print("YAKINŞAMA ANALİZİ")
        print("="*80)
        
        # İlk 10 trial vs son 10 trial
        first_10_avg = df['fitness'].head(10).mean()
        last_10_avg = df['fitness'].tail(10).mean()
        improvement = ((first_10_avg - last_10_avg) / first_10_avg) * 100
        
        print(f"\nİlk 10 trial ortalama fitness: {first_10_avg:.2f}")
        print(f"Son 10 trial ortalama fitness: {last_10_avg:.2f}")
        print(f"İyileşme: {improvement:.2f}%")
        
        # En iyi trial hangi sırada bulundu?
        best_trial_index = df['fitness'].idxmin()
        print(f"\nEn iyi çözüm {best_trial_index + 1}. trial'da bulundu")
        print(f"Toplam {self.n_trials} trial'ın %{(best_trial_index + 1) / self.n_trials * 100:.1f}'inde")
        
        print("="*80)
        
        return {
            'first_10_avg': first_10_avg,
            'last_10_avg': last_10_avg,
            'improvement': improvement,
            'best_trial_index': best_trial_index
        }

# ==================== SENSİTİVİTE ANALİZİ ====================
class SensitivityAnalyzer:
    """Parametre sensitivity analizi"""
    
    def __init__(self, problem, base_params):
        self.problem = problem
        self.base_params = base_params
        self.sensitivity_results = {}
        
    def test_parameter(self, param_name, param_values, n_runs=5):
        """Bir parametreyi farklı değerlerle test et"""
        print(f"\n🔬 '{param_name}' parametresi test ediliyor...")
        print(f"   Test değerleri: {param_values}")
        
        results = []
        
        for value in param_values:
            fitness_scores = []
            times = []
            
            for run in range(n_runs):
                # Parametreleri ayarla
                test_params = self.base_params.copy()
                
                if param_name in ['pop_size', 'generations', 'crossover_rate', 'mutation_rate']:
                    test_params['ga_params'] = test_params.get('ga_params', {}).copy()
                    test_params['ga_params'][param_name] = value
                else:
                    test_params['ts_params'] = test_params.get('ts_params', {}).copy()
                    test_params['ts_params'][param_name] = value
                
                # Hibrit çalıştır
                hybrid = HybridGATS(self.problem, **test_params, verbose=False)
                _, fitness, time, _, _, _ = hybrid.run()
                
                fitness_scores.append(fitness)
                times.append(time)
            
            # Ortalama ve std
            avg_fitness = np.mean(fitness_scores)
            std_fitness = np.std(fitness_scores)
            avg_time = np.mean(times)
            
            results.append({
                'value': value,
                'avg_fitness': avg_fitness,
                'std_fitness': std_fitness,
                'avg_time': avg_time,
                'all_fitness': fitness_scores
            })
            
            print(f"   {param_name}={value}: Fitness={avg_fitness:.2f} (±{std_fitness:.2f}), Süre={avg_time:.2f}s")
        
        self.sensitivity_results[param_name] = results
        return results
    
    def run_full_analysis(self):
        """Tüm parametreler için sensitivity analizi"""
        print("\n" + "="*80)
        print("SENSİTİVİTE ANALİZİ BAŞLIYOR")
        print("="*80)
        
        # Test edilecek parametreler ve değerleri
        test_params = {
            'pop_size': [50, 75, 100, 125, 150],
            'mutation_rate': [0.1, 0.15, 0.2, 0.25, 0.3],
            'crossover_rate': [0.6, 0.7, 0.8, 0.9],
            'tabu_tenure': [2, 4, 6, 8, 10]
        }
        
        for param_name, param_values in test_params.items():
            self.test_parameter(param_name, param_values, n_runs=3)
        
        print("\n" + "="*80)
        print("SENSİTİVİTE ANALİZİ TAMAMLANDI!")
        print("="*80)
        
        return self.sensitivity_results
    
    def get_best_values(self):
        """Her parametre için en iyi değeri bul"""
        print("\n" + "="*80)
        print("PARAMETRELERİN EN İYİ DEĞERLERİ")
        print("="*80)
        
        best_values = {}
        
        for param_name, results in self.sensitivity_results.items():
            best_result = min(results, key=lambda x: x['avg_fitness'])
            best_values[param_name] = best_result['value']
            
            print(f"\n{param_name}:")
            print(f"  En iyi değer: {best_result['value']}")
            print(f"  Fitness: {best_result['avg_fitness']:.2f} (±{best_result['std_fitness']:.2f})")
        
        print("="*80)
        
        return best_values