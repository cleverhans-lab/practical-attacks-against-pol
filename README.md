# Practical attacks against PoL

```
# For Infinitesimal Update Attack 
python infinitesimal_update_attack.py --dataset CIFAR10 --model resnet20 --final-model <path/to/model>  --save-dir inf_update_CIFAR10
python infinitesimal_update_attack.py --dataset CIFAR100 --model resnet50 --final-model <path/to/model> --save-dir inf_update_CIFAR100

# For Blindfold Top-Q Attack
python blindfold_top_q_attack.py --dataset CIFAR10 --model resnet20 --final-model <path/to/model>  --save-dir blindfold_top_q_CIFAR10
python blindfold_top_q_attack.py --dataset CIFAR100 --model resnet50 --final-model <path/to/model> --save-dir blindfold_top_q_CIFAR100
```