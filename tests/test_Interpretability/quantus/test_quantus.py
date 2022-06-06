
import sys
from catboost import train
sys.path.append("/Users/zhanghangsheng/Documents/my_code/Cyber-Security-ML-Toolbox")
from csmt.get_model_data import get_datasets,parse_arguments,models_train,print_results,models_predict,models_load
import numpy as np
import torch
import matplotlib.pyplot as plt
from csmt.Interpretability.captum.attr import *
import csmt.Interpretability.quantus as quantus
import random
import pandas as pd

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
# 设置随机数种子
setup_seed(20)

if __name__=='__main__':
     
     arguments = sys.argv[1:]
     options = parse_arguments(arguments)
     datasets_name=options.datasets
     orig_models_name=options.algorithms

     X_train,y_train,X_val,y_val,X_test,y_test,mask=get_datasets(options)
     X_train,y_train,X_test,y_test=X_train,y_train,X_test,y_test


     # trained_models=models_train(datasets_name,orig_models_name,False,X_train,y_train,X_val,y_val)
     trained_models=models_load(datasets_name,orig_models_name)
     y_test,y_pred=models_predict(trained_models,X_test,y_test)

     table=print_results(datasets_name,orig_models_name,y_test,y_pred,'original_accuracy')

     model=trained_models[0].classifier.model

     x_batch = torch.tensor(X_test[0:100]).float()
     y_batch = torch.tensor(y_test[0:100]).long()

     # print(x_batch.dtype)
     # print(y_batch.dtype)

     a_batch_saliency = quantus.normalise_by_negative(Saliency(model).attribute(inputs=x_batch, target=y_batch, abs=True).sum(axis=1).cpu().numpy())
     a_batch_intgrad = quantus.normalise_by_negative(IntegratedGradients(model).attribute(inputs=x_batch, target=y_batch, baselines=torch.zeros_like(x_batch)).sum(axis=1).cpu().numpy())

     # Save x_batch and y_batch as numpy arrays that will be used to call metric instances.
     x_batch, y_batch = x_batch.cpu().numpy(), y_batch.cpu().numpy()

     # Plot explanations!
     nr_images = 3
     fig, axes = plt.subplots(nrows=nr_images, ncols=3, figsize=(nr_images*2.5, int(nr_images*3)))
     for i in range(nr_images):
          axes[i, 0].imshow((np.reshape(x_batch[i], (28, 28)) * 255).astype(np.uint8), vmin=0.0, vmax=1.0, cmap="gray")
          axes[i, 0].title.set_text(f"MNIST digit {y_batch[i].item()}")
          axes[i, 0].axis("off")
          axes[i, 1].imshow(a_batch_saliency[i], cmap="seismic")
          axes[i, 1].title.set_text(f"Saliency")
          axes[i, 1].axis("off")
          a = axes[i, 2].imshow(a_batch_intgrad[i], cmap="seismic")
          axes[i, 2].title.set_text(f"Integrated Gradients")
          axes[i, 2].axis("off")
     plt.tight_layout()
     plt.show()
     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
     params_eval = {
          "nr_samples": 10,
          "perturb_radius": 0.1,
          "norm_numerator": quantus.fro_norm,
          "norm_denominator": quantus.fro_norm,
          "perturb_func": quantus.uniform_noise,
          "similarity_func": quantus.difference,
          "disable_warnings": True,
     }

     # Return max sensitivity scores in an one-liner - by calling the metric instance.
     # scores_saliency = quantus.MaxSensitivity(**params_eval)(model=model, 
     #      x_batch=x_batch,
     #      y_batch=y_batch,
     #      a_batch=a_batch_saliency,
     #      **{"explain_func": quantus.explain, "method": "Saliency", "device": device, "img_size": 28, "normalise": False, "abs": False})

     # # Return max sensitivity scores in an one-liner - by calling the metric instance.
     # scores_intgrad = quantus.MaxSensitivity(**params_eval)(model=model, 
     # x_batch=x_batch,
     # y_batch=y_batch,
     # a_batch=a_batch_intgrad,
     # **{"explain_func": quantus.explain, "method": "IntegratedGradients", "device": device, "img_size": 28, "normalise": False, "abs": False})

     # print(f"max-Sensitivity scores by Yeh et al., 2019\n" \
     #      f"\n • Saliency = {np.mean(scores_saliency):.2f} ({np.std(scores_saliency):.2f})." \
     #      f"\n • Integrated Gradients = {np.mean(scores_intgrad):.2f} ({np.std(scores_intgrad):.2f})."
     #      )

     # metrics = {"max-Sensitivity": quantus.MaxSensitivity(**params_eval)}


     metrics = {"max-Sensitivity": quantus.MaxSensitivity(**params_eval)}

     xai_methods = {"Saliency": a_batch_saliency,
                    "IntegratedGradients": a_batch_intgrad}

     results = quantus.evaluate(metrics=metrics,
                              xai_methods=xai_methods,
                              model=model,
                              x_batch=x_batch,
                              y_batch=y_batch,
                              agg_func=np.mean,
                              **{"explain_func": quantus.explain, "device": device, "img_size": 28, "normalise": False, "abs": False})

     df = pd.DataFrame(results)
     print(df)



     
    
 









