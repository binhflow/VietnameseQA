from tensorflow import keras
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import run_classifier
import run_classifier_with_tfhub
import tensorflow as tf
import json
import datetime

def download_data(link, name):
  dataset = tf.keras.utils.get_file(
      fname=name+".zip", 
      origin=link, 
      extract=True)
  with open(os.path.join(os.path.dirname(dataset), name+'.json'), "r") as f:
    data_dict = json.load(f)
  return data_dict

def preprocess(data_dict, modelhub):
  df = pd.DataFrame.from_dict(data_dict)
  df['label'] = np.where(df['label']==True,1,0)
  train, dev = train_test_split(df, test_size=0.01, random_state=42)
  train_InputExamples = train.apply(lambda x: run_classifier.InputExample(guid=None,
                                                                   text_a = x['question'], 
                                                                   text_b = x['text'], 
                                                                   label = x['label']), axis = 1)
  dev_InputExamples = dev.apply(lambda x: run_classifier.InputExample(guid=None, 
                                                                   text_a = x['question'], 
                                                                   text_b = x['text'], 
                                                                   label = x['label']), axis = 1)
  tokenizer = run_classifier_with_tfhub.create_tokenizer_from_hub_module(BERT_MODEL_HUB)
  train_features = run_classifier.convert_examples_to_features(
      train_InputExamples, label_list, MAX_SEQ_LENGTH, tokenizer)
  dev_features = bert.run_classifier.convert_examples_to_features(dev_InputExamples, label_list, MAX_SEQ_LENGTH, tokenizer)
  return train_features, dev_features

def model_train(estimator, train_features):
  print('***** Started training at {} *****'.format(datetime.datetime.now()))
  print('  Num examples = {}'.format(len(train)))
  print('  Batch size = {}'.format(TRAIN_BATCH_SIZE))
  tf.logging.info("  Num steps = %d", num_train_steps)
  train_input_fn = run_classifier.input_fn_builder(
      features=train_features,
      seq_length=MAX_SEQ_LENGTH,
      is_training=True,
      drop_remainder=True)
  estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
  print('***** Finished training at {} *****'.format(datetime.datetime.now()))

def model_eval(estimator, dev_features):
  # Eval the model.
  print('***** Started evaluation at {} *****'.format(datetime.datetime.now()))
  print('  Num examples = {}'.format(len(dev)))
  print('  Batch size = {}'.format(EVAL_BATCH_SIZE))
  eval_steps = int(len(dev) / EVAL_BATCH_SIZE)
  eval_input_fn = run_classifier.input_fn_builder(
      features=eval_features,
      seq_length=MAX_SEQ_LENGTH,
      is_training=False,
      drop_remainder=True)
  result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)
  print('***** Finished evaluation at {} *****'.format(datetime.datetime.now()))
  output_eval_file = os.path.join(OUTPUT_DIR, "eval_results.txt")
  with open(output_eval_file, "w") as writer:
    print("***** Eval results *****")
    for key in sorted(result.keys()):
      print('  {} = {}'.format(key, str(result[key])))
      writer.write("%s = %s\n" % (key, str(result[key])))

OUTPUT_DIR ='./wikiqa/outputs'
BERT_MODEL = 'multi_cased_L-12_H-768_A-12'
BERT_MODEL_HUB = 'https://tfhub.dev/google/bert_' + BERT_MODEL + '/1'
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 8
PREDICT_BATCH_SIZE = 8
LEARNING_RATE = 2e-5
NUM_TRAIN_EPOCHS = 3.0
MAX_SEQ_LENGTH = 256
# Warmup is a period of time where hte learning rate 
# is small and gradually increases--usually helps training.
WARMUP_PROPORTION = 0.1
# Model configs
SAVE_CHECKPOINTS_STEPS = 1000
SAVE_SUMMARY_STEPS = 1000
label_list = [0,1]

def main():
  #Global
  train_dict = download_data('https://dl.challenge.zalo.ai/ZAC2019_VietnameseWikiQA/train.zip','train')
  train_features, dev_features = preprocess(train_dict, BERT_MODEL_HUB)
  num_train_steps = int(len(train_features) / TRAIN_BATCH_SIZE * NUM_TRAIN_EPOCHS)
  num_warmup_steps = int(num_train_steps * WARMUP_PROPORTION)
  os.environ['TFHUB_CACHE_DIR'] = OUTPUT_DIR
  run_config = tf.estimator.RunConfig(
    model_dir=OUTPUT_DIR,
    save_summary_steps=SAVE_SUMMARY_STEPS,
    save_checkpoints_steps=SAVE_CHECKPOINTS_STEPS)
  model_fn = models.model_fn_builder(
    num_labels=len(label_list),
    learning_rate=LEARNING_RATE,
    num_train_steps=num_train_steps,
    num_warmup_steps=num_warmup_steps,
    bert_hub_module_handle=BERT_MODEL_HUB
  )
  estimator = tf.estimator.Estimator(
  model_fn=model_fn,
  config=run_config,
  params={"batch_size": BATCH_SIZE})
  train_features, dev_features = preprocess(train_dict)
  model_train(estimator, train_features)
  model_eval(estimator, dev_features)

if __name__ == '__main__':
  main()


