import sys
sys.path.append("/Users/zhanghangsheng/Documents/my_code/Cyber-Security-ML-Toolbox")
# Generate synthetic data and model
import synmod
output_dir = "tests/Interpretability/anamod/example_synthetic_temporal"
num_features = 10
synthesized_features, X, model = synmod.synthesize(output_dir=output_dir, num_instances=100, seed=100,
                                                    num_features=num_features, fraction_relevant_features=0.5,
                                                    synthesis_type="temporal", sequence_length=20, model_type="regressor")
y = model.predict(X, labels=True)

# Analyze the model
from csmt.Interpretability.anamod import TemporalModelAnalyzer
analyzer = TemporalModelAnalyzer(model, X, y, output_dir=output_dir)
features = analyzer.analyze()

# Visualize feature importance for temporal windows
import subprocess
subprocess.run(["open", f"{output_dir}/feature_importance_windows.png"], check=True)