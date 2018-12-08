
import math

import skimage.io
import skimage.transform

import tensorflow as tf
import tensorflow.contrib.microsoft as mstf
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages

from param import FLAGS

def read_dict(dictfile):
    dict = {}
    with open(dictfile) as f:
        for id, line in enumerate(f):
            key = line.rstrip('\n')
            value = id
            dict[key] = value
    return dict

def parse_dims(dims_str):
    dims = []
    for dim in dims_str.split(','):
        dims.append(int(dim))
    return dims

def TextExtract(text, win_size, dict_handle, weight, dim_input, dim_output, max_term_count = 12):
    indices, ids, values, offsets = mstf.dssm_xletter(input=text, win_size=win_size, dict_handle=dict_handle, max_term_count = max_term_count)
#    indices = tf.Print(indices, [indices],'Out[indices]=',summarize = 256)
#    ids = tf.Print(ids, [ids],'Out[ids]=',summarize = 256)
#    offsets = tf.Print(offsets, [offsets],'Out[offsets]=',summarize = 128)
#    print('shape indices:',indices.get_shape())
#    print('shape ids:',ids.get_shape())
#    print('shape values:',values.get_shape())
#    print('shape offsets:',offsets.get_shape())
    offsets_to_dense = tf.segment_sum(tf.ones_like(offsets),offsets)
    batch_id = tf.cumsum(offsets_to_dense[:-1]) #dense offset lei jia
    index_tensor = tf.concat([tf.expand_dims(batch_id,axis=-1), tf.expand_dims(indices,axis=-1)],axis=-1)
#    index_tensor = tf.Print(index_tensor,[index_tensor],'Out[index_tensor]=',summarize = 256)
    value_tensor = ids
#    value_tensor = tf.Print(value_tensor,[value_tensor],'Out[value_tensor]=',summarize = 256)
    dense_shape = tf.concat([tf.shape(offsets),tf.expand_dims(tf.reduce_max(indices) + 1,axis=-1)],axis=0)
#    dense_shape = tf.Print(dense_shape,[dense_shape],'Out[dense_shape]=')
    text_tensor = tf.SparseTensor(indices=tf.cast(index_tensor,tf.int64), values = value_tensor, dense_shape=tf.cast(dense_shape,tf.int64))
    
    text_padding = tf.reduce_max(indices) + 1

    text_tensor = tf.sparse_reshape(text_tensor,[-1])
    text_tensor,text_mask = tf.sparse_fill_empty_rows(text_tensor,dim_input-1)
    text_vecs = tf.nn.embedding_lookup_sparse(weight,text_tensor,None,combiner='sum')
    text_vecs = tf.transpose(tf.multiply(tf.transpose(text_vecs), 1-tf.cast(text_mask,dtype=tf.float32)))
    print('shape text_vecs:',text_vecs.get_shape())
    #return vecs
    #q_vecs = tf.sparse_tensor_dense_matmul(text_tensor, weight_q)
    text_vecs = tf.reshape(text_vecs,[-1,text_padding, dim_output])
    step_mask = tf.equal(tf.reduce_sum(text_vecs,axis=2),0)
    step_mask = tf.where(step_mask ,-math.inf*tf.ones_like(step_mask,dtype=tf.float32),tf.zeros_like(step_mask,dtype=tf.float32))
    return text_vecs, text_padding, step_mask
    
def Maxpooling(text_vecs, text_padding,step_mask, dim_input, dim_output):
    vecs_4maxpool = text_vecs + tf.expand_dims(step_mask,axis=-1)
    maxpool = tf.reduce_max(vecs_4maxpool, axis=1)
    maxpool = tf.where(tf.is_finite(maxpool), maxpool, tf.zeros_like(maxpool))
    return maxpool
    
def KMaxpooling(text_vecs, text_padding, step_mask, dim_input, dim_output, k_max_pooling = 1):
    vecs_4maxpool = text_vecs + tf.expand_dims(step_mask,axis=-1)
    vecs_4maxpool = tf.transpose(vecs_4maxpool,perm=[0,2,1])
    maxpool,_ = tf.nn.top_k(vecs_4maxpool, k_max_pooling)
    maxpool = tf.reshape(maxpool, tf.cast([-1, dim_output * k_max_pooling],tf.int64))
    maxpool = tf.where(tf.is_finite(maxpool), maxpool, tf.zeros_like(maxpool))
    print("shape maxpool:", maxpool.get_shape())
    return maxpool
    
def Conv(name, x, filter_width, in_filters, out_filters, strides):
      n = filter_width * out_filters
      kernel = tf.get_variable(
              name, 
              [ filter_width, in_filters, out_filters],
              tf.float32, 
              initializer=tf.random_normal_initializer(stddev=math.sqrt(2.0/n)))
      return tf.nn.conv1d(x, kernel, strides, padding='SAME') 

# ((x-mean)/var)*gamma+beta
def batch_norm(name, x, is_training):  
    x_shape = x.get_shape()  
    params_shape = x_shape[-1:]  
  
    axis = list(range(len(x_shape) - 1))  
  
    beta = tf.get_variable(name + 'beta', params_shape, initializer=tf.zeros_initializer())  
    gamma = tf.get_variable(name + 'gamma', params_shape, initializer=tf.ones_initializer())  
  
    moving_mean = tf.get_variable(name + 'moving_mean', params_shape, initializer=tf.zeros_initializer(), trainable=False)  
    moving_variance = tf.get_variable(name + 'moving_variance', params_shape, initializer=tf.ones_initializer(), trainable=False)  
  
    # These ops will only be preformed when training.  
    mean, variance = tf.nn.moments(x, axis)  
    update_moving_mean = moving_averages.assign_moving_average(moving_mean, mean, 0.9997)  
    update_moving_variance = moving_averages.assign_moving_average(moving_variance, variance, 0.9997)  
    tf.add_to_collection('resnet_update_ops', update_moving_mean)  
    tf.add_to_collection('resnet_update_ops', update_moving_variance)  
  
    mean, variance = control_flow_ops.cond(  
        is_training, lambda: (mean, variance),  
        lambda: (moving_mean, moving_variance))  
  
    return tf.nn.batch_normalization(x, mean, variance, beta, gamma, 0.001)
    
def SampleResidual(name, x, filter_width, in_filters, activate_before_residual=False, is_BN = False, is_training = True):
    if activate_before_residual:
        with tf.variable_scope('shared_activation'):
            if is_BN:
                x = batch_norm(name, x, tf.constant(is_training))
            x = tf.nn.relu(x)
            orig_x = x
    else:
        with tf.variable_scope('residual_only_activation'):
            orig_x = x
            if is_BN:
                x = batch_norm(name, x, tf.constant(is_training))
            x = tf.nn.relu(x)
        
    with tf.variable_scope('sub1'):
        x = Conv(name + "Conv1", x,  filter_width, in_filters, in_filters, 1)
      
    with tf.variable_scope('sub2'):
        if is_BN:
            x = batch_norm(name, x, tf.constant(is_training))
        x = tf.nn.relu(x)
        x = Conv(name + "Conv2", x, filter_width, in_filters , in_filters, 1)
      
    with tf.variable_scope('sub_add'):
        x += orig_x
        
    return x
    
def dssm_xletter_conv(text, win_size, dict_handle, weight, max_term_count = 12):
    indices, ids, values, offsets = mstf.dssm_xletter(input=text, win_size=win_size, dict_handle=dict_handle, max_term_count = max_term_count)
#    indices = tf.Print(indices, [indices],'Out[indices]=',summarize = 256)
#    ids = tf.Print(ids, [ids],'Out[ids]=',summarize = 256)
#    offsets = tf.Print(offsets, [offsets],'Out[offsets]=',summarize = 128)
    print('shape indices:',indices.get_shape())
    print('shape ids:',ids.get_shape())
    print('shape values:',values.get_shape())
    print('shape offsets:',offsets.get_shape())
    
    return mstf.dssm_conv(input_indices=indices,
                          input_ids=ids,
                          input_values=values,
                          input_offsets=offsets,
                          weight=weight,
                          max_pooling=True)
    
def dssm_xletter_conv2(text, win_size, dict_handle, weight, dim_input, dim_output, max_term_count = 12, k_max_pooling = 1):  
    text_vecs, text_padding, step_mask = TextExtract(text, win_size, dict_handle, weight, dim_input, dim_output, max_term_count)
    if k_max_pooling == 1:
        maxpool = Maxpooling(text_vecs, text_padding, step_mask, dim_input, dim_output)
    else:
        maxpool = KMaxpooling(text_vecs, text_padding, step_mask, dim_input, dim_output, k_max_pooling)
    return maxpool
    
def dssm_xletter_conv3(text, win_size, dict_handle, weight, dim_input, dim_output, max_term_count = 12, k_max_pooling = 1, name = 'q_conv_', is_BN = False, is_training = True):  
    text_vecs, text_padding, step_mask = TextExtract(text, win_size, dict_handle, weight, dim_input, dim_output, max_term_count)
    text_vecs = SampleResidual(name, text_vecs, 3, dim_output, False, is_BN, is_training)
    if k_max_pooling == 1:
        maxpool = Maxpooling(text_vecs, text_padding, step_mask, dim_input, dim_output)
    else:
        maxpool = KMaxpooling(text_vecs, text_padding, step_mask, dim_input, dim_output, k_max_pooling)
    return maxpool

def ICEEmb(ice, weight):
    ices = tf.string_split(ice, delimiter=',')
    ice_indices = tf.reshape(tf.slice(ices.indices,[0,0],[-1,1]),[-1,1])
    ice_values = tf.string_to_number(ices.values,out_type = tf.int64)
    ice_tensor = tf.SparseTensor(indices=ice_indices, values = ice_values, dense_shape=tf.cast([tf.reduce_max(ice_indices) + 1], tf.int64))
    ice_emb = tf.nn.embedding_lookup_sparse(weight, ice_tensor, None,combiner='sum')
    return ice_emb
    
def huber_loss(labels, predictions, delta=1.0):
   residual = tf.abs(predictions - labels)
   condition = tf.less(residual, delta)
   small_res = 0.5 * residual**2
   large_res = delta * residual - 0.5 * delta**2
   return tf.where(condition, small_res, large_res)

def get_activation_func(activation):
  if activation == 'relu':
    return tf.nn.relu
  elif activation == 'tanh':
    return tf.nn.tanh
  elif activation == 'sigmoid':
    return tf.nn.sigmoid
  else:
    raise ValueError('Unkown activation: {}'.format(activation))

class CDSSMModel(mstf.Model):
    def __init__(self, dict_file, dims, win_size, negative_count, is_training=True,
                 hidden_activation='relu', last_activation='tanh', last_bn=True, softmax_gamma=10):
        self.dict_file = dict_file
        self.dim_dict = len(read_dict(dict_file))
        self.dims = dims
        self.win_size = win_size
        self.negative_count = negative_count
        self.is_training = is_training
        self.hidden_activation_func = get_activation_func(hidden_activation.lower())
        self.last_activation_func = get_activation_func(last_activation.lower())
        self.last_bn = last_bn
        self.softmax_gamma = softmax_gamma

    def calc_loss(self, pred, label):
        loss = 0.0
        if FLAGS.lossfuc == "mse":
            loss = tf.reduce_sum(tf.pow(pred-label, 2))
        elif FLAGS.lossfuc == "log":
            loss = -tf.reduce_sum(label * tf.log(tf.clip_by_value((pred+1.0)/2.0, 1e-10, 1.0)) + (1-label) * tf.log(tf.clip_by_value(1-pred, 1e-10, 1.0)))
        else:
            loss = tf.reduce_sum(tf.pow(pred-label, 2), weight)
        tf.summary.scalar('losses',loss)
        return loss

    def build_graph(self):
        query = tf.placeholder(tf.string, [None], name='query')
        keyword = tf.placeholder(tf.string, [None], name='keyword')
        
        label = tf.placeholder(tf.float32, [None], name='label')
        #qice = tf.placeholder(tf.string,[None],name = 'qice')
        #kice = tf.placeholder(tf.string,[None],name = 'kice')
        
        op_dict = mstf.dssm_dict(self.dict_file)

        dim_input = self.dim_dict * self.win_size
        

        for i, dim in enumerate(self.dims):
            dim_output = dim
            random_range = math.sqrt(6.0 / (dim_input + dim_output))

            with tf.variable_scope("layer{:}".format(i)):
                weight_q = tf.get_variable(name='weight_q',
                                           shape=[dim_input, dim_output],
                                           initializer=tf.random_uniform_initializer(-random_range, random_range))
                weight_d = tf.get_variable(name='weight_d',
                                           shape=[dim_input, dim_output],
                                           initializer=tf.random_uniform_initializer(-random_range, random_range))

                if i == 0:
                    #random_range_ice = math.sqrt(6.0 / (3181 + 288))
                    #weight_qice_layer = tf.get_variable(name='weight_qice_layer',
                    #                       shape=[3181, 288],
                    #                       initializer=tf.random_uniform_initializer(-random_range_ice, random_range_ice))
                    #weight_kice_layer = tf.get_variable(name='weight_kice_layer',
                    #                       shape=[3181, 288],
                    #                       initializer=tf.random_uniform_initializer(-random_range_ice, random_range_ice))

                    op_output_q, _ = dssm_xletter_conv(query, self.win_size, op_dict, weight_q, max_term_count = 12)
                    print('op_output_q:', op_output_q.get_shape())
                    op_output_d, _ = dssm_xletter_conv(keyword, self.win_size, op_dict, weight_d, max_term_count = 12)
                    
                    #op_output_qice = ICEEmb(qice, weight_qice_layer)
                    #op_output_kice = ICEEmb(kice, weight_kice_layer)
                    
                    #op_output_q = tf.concat([op_output_q, op_output_qice], axis=1)
                    #op_output_d = tf.concat([op_output_d, op_output_kice], axis=1)
                    
                    #dim_input = dim_output * 1 + 288
                    dim_input = dim_output
        
                else:
                    op_output_q = tf.matmul(op_input_q, weight_q)
                    op_output_d = tf.matmul(op_input_d, weight_d)
                    dim_input = dim_output

                if i == len(self.dims) - 1:
                    op_output_q = self.last_activation_func(op_output_q)
                    op_output_d = self.last_activation_func(op_output_d)
                else:
                    op_output_q = self.hidden_activation_func(op_output_q)
                    op_output_d = self.hidden_activation_func(op_output_d)
                 
#                if i != len(self.dims) - 1:
#                    op_output_q = self.hidden_activation_func(op_output_q)
#                    op_output_d = self.hidden_activation_func(op_output_d)

            op_input_q = op_output_q
            op_input_d = op_output_d
#            dim_input = dim_output
        
        #self.last_bn = True
        if self.last_bn:
            op_output_q = tf.contrib.layers.batch_norm(op_output_q, center=True, scale=False, is_training=self.is_training)
            op_output_d = tf.contrib.layers.batch_norm(op_output_d, center=True, scale=False, is_training=self.is_training)

        # # CDSSM softmax implemented with customized operator
        # op_softmax, *_ = mstf.dssm_softmax(query=op_output_q, doc=op_output_d, negative_count=self.negative_count)
        # op_score, *_ = mstf.dssm_softmax(query=op_output_q, doc=op_output_d, negative_count=0)

        # CDSSM softmax implemented with TF build-in operators
        op_output_q = tf.nn.l2_normalize(op_output_q, dim=1)
        op_output_d = tf.nn.l2_normalize(op_output_d, dim=1)
#        scos = tf.reduce_sum(tf.multiply(op_output_q, op_output_d), axis=1)

        W_1 = tf.get_variable('weight_1', [1], initializer=tf.constant_initializer(1.0))
        b_1 = tf.get_variable('b_1', [1], initializer=tf.constant_initializer(0.01))

        cosines = tf.reduce_sum(tf.multiply(op_output_q, op_output_d), axis=1)
        scos = cosines * W_1 + b_1

        #op_loss = tf.reduce_mean(tf.abs(cosines-label)) #mae
        #op_loss = tf.reduce_sum(tf.abs(cosines-label)) #mae
        #op_loss = tf.reduce_sum(tf.pow(scos-label, 2)) #mse
        #op_loss = tf.reduce_sum(weight * tf.pow(cosines-label, 2)) #mse
        #op_loss = tf.reduce_sum(huber_loss(label, cosines, 0.70))
        op_loss = self.calc_loss(scos, label)
        

        return {'train': mstf.Model.Action({'loss': (op_loss, True), 'score': cosines}, {'query': query, 'keyword': keyword, 'label':label}),
                'predict': mstf.Model.Action({'score': cosines}, {'query': query, 'keyword': keyword})}

