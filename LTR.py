import numpy as nump
import pandas as pand
import pyterrier as pter
from sklearn.ensemble import RandomForestRegressor
from os import path
if not pter.started():
  pter.init()

dataset = pter.get_dataset("trec-deep-learning-passages")
def passage_read_generator():
  with pter.io.autoopen(dataset.get_corpus()[0], 'rt') as corpusfile:
    for l in corpusfile:
        docno, passage = l.split("\t")
        yield {'docno' : docno, 'text' : passage}


iter_indexer = pter.IterDictIndexer("./trec_index", verbose=True)
print("post iter_indexer")

if path.exists('./trec_index/data.properties'):
  print("reading index from dir")
  index = pter.IndexFactory.of("./trec_index/data.properties")
else:
  indexref = iter_indexer.index(passage_read_generator(), meta=['docno', 'text'],  meta_lengths=[20, 4096])
  print("post indexref")
  index = pter.IndexFactory.of(indexref)

print("post index")

DPH_br = pter.BatchRetrieve(index, wmodel="DPH") 
BM25_br = pter.BatchRetrieve(index, wmodel="BM25") 

FBR = pter.FeaturesBatchRetrieve(index, controls = {"wmodel": "BM25"}, 
  features=["SAMPLE", "WMODEL:TF_IDF", "WMODEL:PL2", "WMODEL:IFB2", "WMODEL:Hiemstra_LM", "WMODEL:LGD", "WMODEL:DPH", "WMODEL:In_expC2", "WMODEL:ML2"])#, "WMODEL:MRFDependenceScoreModifier"])

topics_train = dataset.get_topics('train')
topics_test = dataset.get_topics('test-2019')
qrels_train = dataset.get_qrels('train')
qrels_test = dataset.get_qrels('test-2019')

print("Starting pipeline")
BaselineLTR = FBR >> pter.ltr.apply_learned_model(RandomForestRegressor(n_estimators=200, n_jobs=-1, max_depth=10), form='regression')
print("Starting fitting")
BaselineLTR.fit(topics_train, qrels_train) # Should use a subsample from training dataset instead

print("Starting experiments")
results = pter.pipelines.Experiment([DPH_br, BM25_br, BaselineLTR], topics_test, qrels_test, names=["DPH Baseline", "BM25", "LTR Baseline"], eval_metrics=["trec_eval"])
results

# this runs an experiment to obtain results on the TREC 2019 Deep Learning track queries and qrels
#pter.Experiment([DPH_br, BM25_br], dataset.get_topics("test-2019"), dataset.get_qrels("test-2019"), eval_metrics=["recip_rank", "map"])

