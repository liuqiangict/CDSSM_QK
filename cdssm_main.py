import os
import sys
import glob
import random
import logging
import numpy as np

from param import FLAGS

import tensorflow as tf
import tensorflow.contrib.microsoft as mstf
import pandas as pd

from sklearn.metrics import roc_auc_score

np.set_printoptions(threshold=np.nan, linewidth=np.nan)

from dssm_lib import *
    
if __name__ == '__main__':
    scriptdir = os.path.dirname(os.path.realpath(__file__))

    has_validation = FLAGS.input_validation_data_path is not None

    initial_model = FLAGS.input_previous_model_path
    training_files = mstf.DataSource.get_files_to_read(FLAGS.input_training_data_path)
    validation_files = mstf.DataSource.get_files_to_read(FLAGS.input_validation_data_path)
    dict_file = os.path.join(scriptdir, FLAGS.dict_path)
    model_dir = FLAGS.output_model_path
    save_model_every_epoch = FLAGS.save_model_every_epoch

    epochs = FLAGS.epochs
    learning_rate = FLAGS.learning_rate
    batch_size = FLAGS.batch_size
    dims = parse_dims(FLAGS.dims)
    win_size = FLAGS.win_size
    negative_count = FLAGS.negative_count
    thread_count = FLAGS.thread_count
    batch_num_to_print_loss = FLAGS.batch_num_to_print_loss

    training_data_schema = FLAGS.training_data_schema
    validation_data_schema = FLAGS.validation_data_schema
    training_data_shuffle = FLAGS.training_data_shuffle

    save_predict_result = FLAGS.save_predict_result
    hidden_activation = FLAGS.hidden_activation
    last_activation = FLAGS.last_activation
    last_bn = FLAGS.last_bn
    convert_model = FLAGS.convert_model
    
    logging.basicConfig(level=logging.INFO,
                        format='[%(levelname)s %(asctime)s] %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')

    logging.info('Training.....')
    
    logging.info('training_files: {}'.format(training_files))
    logging.info('validation_files: {}'.format(validation_files))
    logging.info('training_data_schema: {}'.format(training_data_schema))
    logging.info('validation_data_schema: {}'.format(validation_data_schema))
    logging.info('training_data_shuffle: {}'.format(training_data_shuffle))
    logging.info('dict: {}'.format(dict_file))
    logging.info('model_dir: {}'.format(model_dir))
    logging.info('epochs: {}'.format(epochs))
    logging.info('learning_rate: {}'.format(learning_rate))
    logging.info('batch_size: {}'.format(batch_size))
    logging.info('dims: {}'.format(dims))
    logging.info('win_size: {}'.format(win_size))
    logging.info('negative_count: {}'.format(negative_count))
    logging.info('thread_count: {}'.format(thread_count))
    logging.info('batch_num_to_print_loss: {}'.format(batch_num_to_print_loss))
    logging.info('hidden_activation: {}'.format(hidden_activation))
    logging.info('last_activation: {}'.format(last_activation))
    logging.info('last_bn: {}'.format(last_bn))

    setting = mstf.RunSetting(sys.argv[1:])

    model = CDSSMModel(dict_file, dims, win_size, negative_count, hidden_activation=hidden_activation, last_activation=last_activation, last_bn=last_bn)

    def process_label_test(label):
        return label
        
    def process_label_train(label):
        return float(label)
        
    def process_weight_train(weight):
        return float(weight)

    def share_info(x):
        return x
    
    training_ds = mstf.TextDataSource(training_files, batch_size, training_data_schema, shuffle=training_data_shuffle, post_process={'label': process_label_train, 'qice':share_info, 'kice':share_info})
    validation_ds = mstf.TextDataSource(validation_files, batch_size, validation_data_schema, post_process={'label': process_label_test, 'id':share_info, 'qice':share_info, 'kice':share_info})
    
    trainer = mstf.Trainer.create_trainer(setting)
    trainer.optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    
    trainer.setup(model)
    trainer.start()

    #convert model
    if convert_model:
        for t_vars in trainer._train_vars:
            for v in t_vars:
                weight = trainer._sess.run(v)
                logging.info("Convert {} to TSV...".format(v.name))
                outputW = tf.gfile.GFile(model_dir + '/' + v.name.replace('/','_').replace(':','_'), mode='w')
                if len(np.shape(weight)) == 1:
                    weight = weight.reshape(-1,1)
                outputW.write("{} {}".format(len(weight),len(weight[0])) + "\n")
                for lweight in weight:
                    first = True
                    for w in lweight:
                        if(not first):
                            outputW.write(" ")
                        outputW.write("{:.8f}".format(w))
                        first = False
                    outputW.write("\n")
                outputW.close()
        sys.exit(0)

    if setting.job_name == 'worker':
        for epoch in range(epochs):
            #os.chdir(exe_path)
            
            total_loss = 0.0
            total_samples = 0
            total_loss_avg = 0
            i = 0

            for _, result, batch, samples in trainer.run(training_ds, 'train'):
                loss = result['loss']

                total_loss += loss
                total_samples += samples
                partial_loss_avg = loss / samples
                total_loss_avg = total_loss / total_samples

                if (i % batch_num_to_print_loss == 0):
                    logging.info('Batch: {:}, partial average loss: {:f}, total average loss: {:f}'.format(i, partial_loss_avg, total_loss_avg))
                    
                i += 1

            logging.info('Epoch {} finished, total average loss: {:f}'.format(epoch, total_loss_avg))

            if has_validation:
                scores = []
                labels = []

                for _, result, batch, _ in trainer.run(validation_ds, 'predict'):
                    scores.extend(result['score'])
                    labels.extend(batch['label'])

                rocauc = roc_auc_score(labels, scores)
                logging.info('AUC: '+ str(rocauc))

            logging.info('PROGRESS: {:.2f}%'.format(100 * epoch / epochs))

            if model_dir is not None and save_model_every_epoch:
                trainer.save_model(os.path.join(model_dir, 'model_{}'.format(epoch)))

        if model_dir is not None:
            trainer.save_model(os.path.join(model_dir, 'model_final'))
            
        #save predict results
        if has_validation and save_predict_result:
            ids = []
            queries = []
            keywords = []
            labels = []
            pscores = []
            outputter = tf.gfile.GFile(model_dir + "/pred.txt", mode='w')
            for _, result, batch, _ in trainer.run(validation_ds, 'predict'):
                pscores.extend(result['score'])
                ids.extend(batch['id'])
                queries.extend(batch['query'])
                keywords.extend(batch['keyword'])
                labels.extend(batch['label'])				
            for i in range(0, len(ids)):
                outputter.write(ids[i] + "\t" + queries[i] + "\t" + keywords[i] + "\t" + labels[i] + "\t" + str(pscores[i].item()) + "\n")
            outputter.close()
            
    trainer.stop()

    logging.info('Done')