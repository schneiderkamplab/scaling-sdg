from datasets import load_dataset
from synthesizers import pipeline
# load a dataset on breast cancer
ds = load_dataset("mstz/breast", split="train")
print(ds.to_pandas())
# set up pipelines with default values
train=pipeline("train") # train a model using synthcity's implementation of adsgans
gen=pipeline("generate") # generate synthtetic data given model and count
eval=pipeline("evaluate", target_col="is_cancer") # evaluate synthetic data using syntheval with target column set to "is_cancer"
# train the model
state=train(ds)
print(state)
# test generation
state = gen(state, 100)
print(state)
# test generation
state = eval(state)
print(state)
# small scaling study
sizes = [2**i for i in range(7, 17)] # from 2^7==128 to 2^(17-1)=16384
results=[eval(gen(state,n)).eval for n in sizes] # run evaluations
print("Synthetic dataset sizes          :", sizes)
print("Attribute disclosure risk F1     :", [r["attr_discl_cats"]["Attr Dis macro F1"] for r in results])
print("Median distance to closes record :", [r["dcr"]["mDCR"] for r in results])
print("Mutual information differences   :", [r["mi_diff"]["mutual_inf_diff"] for r in results])
