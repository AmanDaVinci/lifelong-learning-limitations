# Limitations of existing lifelong learning frameworks

To run the shared head experiment on the Lifelong Text Classification benchmark:
```
python main.py --max_train_size=115000 --head_mode=uni --task_stream="domainshift"
```
To run the multiple head experiment on the Lifelong Text Classification benchmark:
```
python main.py --max_train_size=115000 --head_mode=multi --task_stream="domainshift"
```