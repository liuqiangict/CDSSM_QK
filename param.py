
import tensorflow as tf

tf.app.flags.DEFINE_string('input-previous-model-path', None, 'initial model dir')
tf.app.flags.DEFINE_string('input-training-data-path', 'data/train', 'training data dir')
tf.app.flags.DEFINE_string('input-validation-data-path', 'data/test', 'validation data dir')
tf.app.flags.DEFINE_string('output-model-path', 'model', 'output model dir')
tf.app.flags.DEFINE_bool('save-model-every-epoch', True, 'whether save model every epoch')
tf.app.flags.DEFINE_string('dict-path', 'l3g.txt', 'path of x-letter dict')
tf.app.flags.DEFINE_integer('epochs', 10, 'epochs') 
tf.app.flags.DEFINE_float('learning-rate', 0.0008, 'learning rate')  #0.002
tf.app.flags.DEFINE_string('lossfuc', 'log', 'Loss Function: mse, log.')
tf.app.flags.DEFINE_integer('batch-size', 128, 'batch size')
tf.app.flags.DEFINE_integer('win-size', 3, 'windows size')
tf.app.flags.DEFINE_integer('negative-count', 0, 'number of random negative sample count')
tf.app.flags.DEFINE_string('dims', '288, 64', 'dimensions of each layers') #288, 64
tf.app.flags.DEFINE_integer('thread-count', 30, 'number of threads')
tf.app.flags.DEFINE_integer('batch-num-to-print-loss', 1000, 'number of batch number to print loss')
tf.app.flags.DEFINE_string('training-data-schema', 'query:0,keyword:1,label:2', 'schema of training data')
tf.app.flags.DEFINE_string('validation-data-schema', 'query:0,keyword:1,label:2,id:3', 'schema of validation data')
tf.app.flags.DEFINE_bool('training-data-shuffle', True, 'if shuffle training data') #False
tf.app.flags.DEFINE_string('hidden-activation', 'relu', 'Activation for CDSSM hidden layers')
tf.app.flags.DEFINE_string('last-activation', 'tanh', 'Activation for CDSSM last layer')
tf.app.flags.DEFINE_bool('last-bn', False, 'Batch normalization at last layer')
tf.app.flags.DEFINE_bool('convert-model', False, 'convert model or not')
tf.app.flags.DEFINE_bool('save-predict-result', True, 'whether save predict result')

FLAGS = tf.app.flags.FLAGS