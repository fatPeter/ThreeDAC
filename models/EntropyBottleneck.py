"""

Based on https://github.com/yydlmzyz/PCGCv2/blob/main/models/EntropyModel.py
"""


import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import numpy as np

# import tensorflow.contrib.coder from range codec.
import tensorflow as tf
from tensorflow.contrib.coder.python.ops import coder_ops

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
tf.enable_eager_execution(config=tf_config)



class RoundNoGradient(torch.autograd.Function):
  """ TODO: check. """
  @staticmethod
  def forward(ctx, x):
    return x.round()

  @staticmethod
  def backward(ctx, g):
    return g


# class UniverseQuantR(torch.autograd.Function):
#   @staticmethod
#   def forward(ctx, x):
#     b = np.random.uniform(-1,1)
#     uniform_distribution = torch.distributions.Uniform(-0.5*torch.ones(x.size())*(2**b), 
#                                                         0.5*torch.ones(x.size())*(2**b)).sample().cuda()
#     return torch.round(x+uniform_distribution)-uniform_distribution

#   @staticmethod
#   def backward(ctx, g):
#     return g


class Low_bound(torch.autograd.Function):
  """ TODO: check. """
  @staticmethod
  def forward(ctx, x):
    ctx.save_for_backward(x)
    x = torch.clamp(x, min=1e-8)
    return x

  @staticmethod
  def backward(ctx, g):
    x, = ctx.saved_tensors
    grad1 = g.clone()
    grad1[x<1e-8] = 0
    
    pass_through_if = np.logical_or(x.detach().cpu().numpy() >= 1e-8, g.detach().cpu().numpy()<0.0)
    t = torch.Tensor(pass_through_if+0.0).cuda()
    # t = torch.Tensor(pass_through_if+0.0)
    # pass_through_if = torch.ge(x, 1e-8) | torch.lt(g, 0.0)
    # t = pass_through_if.float()
    return grad1*t




class ConvLayer_parameter(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ConvLayer_parameter, self).__init__()        
        factor = 24
        self.conv1 = torch.nn.Conv1d(in_channel, factor, 1)
        self.conv2 = torch.nn.Conv1d(factor, factor, 1)
        self.conv3 = torch.nn.Conv1d(factor, factor, 1)
        
        self.conv4_1 = torch.nn.Conv1d(factor, out_channel, 1)
        self.conv4_2 = torch.nn.Conv1d(factor, out_channel, 1)
        self.conv4_3 = torch.nn.Conv1d(factor, out_channel, 1)
        
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        x1 = self.conv4_1(x)
        x2 = self.conv4_2(x)
        x3 = self.conv4_3(x)
        
        return x1, x2, x3
    
    
    
    




class EntropyBottleneck(nn.Module):
  """The layer implements a flexible probability density model to estimate
  entropy of its input tensor, which is described in this paper:
  >"Variational image compression with a scale hyperprior"
  > J. Balle, D. Minnen, S. Singh, S. J. Hwang, N. Johnston
  > https://arxiv.org/abs/1802.01436"""
  
  def __init__(self, channels, content_chann, likelihood_bound=1e-8, range_coder_precision=16,
                init_scale=8, filters=(3,3,3)):
      
    self.time_comp_sum=0 
    self.time_cdf=0    
    """
    TODO: get channels from input tensor.
    """

    super(EntropyBottleneck, self).__init__()
    self._likelihood_bound = float(likelihood_bound)
    self._range_coder_precision = int(range_coder_precision)
    self._init_scale = float(init_scale)
    self._filters = tuple(int(f) for f in filters)
    self._channels = channels

    # build.
    filters = (1,) + self._filters + (1,)
    scale = self._init_scale ** (1 / (len(self._filters) + 1))
    
    
    # Create variables.
    self._matrices = nn.ParameterList([])
    self._biases = nn.ParameterList([])
    self._factors = nn.ParameterList([])    
    
    self._factors2 = nn.ParameterList([])    
    

    for i in range(len(self._filters) + 1):
      #
      self.matrix = Parameter(torch.FloatTensor(channels, filters[i + 1], filters[i]))
      init_matrix = np.log(np.expm1(1.0 / scale / filters[i + 1]))
      
      
      
      self.matrix.data.fill_(init_matrix)
      self._matrices.append(self.matrix)
      #
      self.bias = Parameter(torch.FloatTensor(channels, filters[i + 1], 1))
      init_bias = torch.FloatTensor(np.random.uniform(-0.5, 0.5, self.bias.size()))
      self.bias.data.copy_(init_bias)# copy or fill?
      self._biases.append(self.bias)
      #       
      self.factor = Parameter(torch.FloatTensor(channels, filters[i + 1], 1))
      self.factor.data.fill_(0.0)
      self._factors.append(self.factor)
      
      self.factor2 = Parameter(torch.FloatTensor(1,))
      self.factor2.data.fill_(0.0)
      self._factors2.append(self.factor2)      
    
   
    
    
    # content encoding
    self.conv_content1 = ConvLayer_parameter(in_channel=content_chann,out_channel=1)
    self.conv_content2 = ConvLayer_parameter(in_channel=content_chann,out_channel=1)
    self.conv_content3 = ConvLayer_parameter(in_channel=content_chann,out_channel=1)
   
    



  def _logits_cumulative(self, inputs, content):
    """Evaluate logits of the cumulative densities.
    
    Arguments:
      inputs: The values at which to evaluate the cumulative densities,
        expected to have shape `(channels, 1, batch)`.

    Returns:
      A tensor of the same shape as inputs, containing the logits of the
      cumulatice densities evaluated at the the given inputs.
    """
    logits = inputs
    

    content = content.transpose(1,2).transpose(2,0)
 
    
    
    k1, b1, c1 = self.conv_content1(content)
    k2, b2, c2 = self.conv_content2(content)
    k3, b3, c3 = self.conv_content3(content)
    k1, k2, k3 = torch.abs(k1), torch.abs(k2), torch.abs(k3)
    
    
    logits_content = k1 * logits + b1
    logits_content = logits_content + torch.tanh(c1) * torch.tanh(logits_content)
    
    logits_content = k2 * logits_content + b2
    logits_content = logits_content + torch.tanh(c2) * torch.tanh(logits_content)    
    
    logits_content = k3 * logits_content + b3
    logits_content = logits_content + torch.tanh(c3) * torch.tanh(logits_content)    
    
    
  
    
    
    
    for i in range(len(self._filters) + 1):
      matrix = torch.nn.functional.softplus(self._matrices[i])
      logits = torch.matmul(matrix, logits)

      logits += self._biases[i]

      factor = torch.tanh(self._factors[i])
      
      logits += factor * torch.tanh(logits)  
      
      
      
      
      
        
    return torch.abs(self._factors2[0]+1)*logits + torch.abs(self._factors2[1]+0.01)*logits_content





  def _logits_cumulative_compress(self, inputs, content):
    """Evaluate logits of the cumulative densities.
    
    Arguments:
      inputs: The values at which to evaluate the cumulative densities,
        expected to have shape `(channels, 1, batch)`.

    Returns:
      A tensor of the same shape as inputs, containing the logits of the
      cumulatice densities evaluated at the the given inputs.
    """
    logits = inputs
    
    pro_range = inputs.shape[-1]
    num_points = content.shape[0]
    content = content.transpose(1,2).transpose(2,0)
    content = content.repeat_interleave(pro_range, -1)
    
    
    
    # pro_range = inputs.shape[-1]
    # num_points = content.shape[0]
    
   
    
    
    # content = content.transpose(1,0)[None,:]
    # content = content.repeat(logits.shape[0], 1, 1)  
    
    # content = content.repeat_interleave(pro_range, -1)
    
    
    
    
    
    
    
    logits_t = logits.repeat((1,1,num_points))
    
  
    k1, b1, c1 = self.conv_content1(content)
    k2, b2, c2 = self.conv_content2(content)
    k3, b3, c3 = self.conv_content3(content)
    k1, k2, k3 = torch.abs(k1), torch.abs(k2), torch.abs(k3)
    
    
    logits_content = k1 * logits_t + b1
    logits_content = logits_content + torch.tanh(c1) * torch.tanh(logits_content)
    
    logits_content = k2 * logits_content + b2
    logits_content = logits_content + torch.tanh(c2) * torch.tanh(logits_content)    
    
    logits_content = k3 * logits_content + b3
    logits_content = logits_content + torch.tanh(c3) * torch.tanh(logits_content)    
      
    
  
   
    
    
    logits_content = logits_content.reshape((-1,num_points,pro_range))
  
    
    
    
    for i in range(len(self._filters) + 1):
      matrix = torch.nn.functional.softplus(self._matrices[i])
      logits = torch.matmul(matrix, logits)

      logits += self._biases[i]

      factor = torch.tanh(self._factors[i])
      
      logits += factor * torch.tanh(logits)
      
      
        
    return torch.abs(self._factors2[0]+1)*logits + torch.abs(self._factors2[1]+0.01)*logits_content





  def _logits_cumulative_decompress(self, inputs, content, channel_idx):
    """Evaluate logits of the cumulative densities.
    
    Arguments:
      inputs: The values at which to evaluate the cumulative densities,
        expected to have shape `(channels, 1, batch)`.

    Returns:
      A tensor of the same shape as inputs, containing the logits of the
      cumulatice densities evaluated at the the given inputs.
    """
    logits = inputs
    
    pro_range = inputs.shape[-1]
    num_points = content.shape[0]
    content = content.transpose(1,2).transpose(2,0)
    content = content.repeat_interleave(pro_range, -1)
    
    
    
    
    
    
    
    logits_t = logits.repeat((1,1,num_points))
    
  
    k1, b1, c1 = self.conv_content1(content)
    k2, b2, c2 = self.conv_content2(content)
    k3, b3, c3 = self.conv_content3(content)
    k1, k2, k3 = torch.abs(k1), torch.abs(k2), torch.abs(k3)
    
    
    logits_content = k1 * logits_t + b1
    logits_content = logits_content + torch.tanh(c1) * torch.tanh(logits_content)
    
    logits_content = k2 * logits_content + b2
    logits_content = logits_content + torch.tanh(c2) * torch.tanh(logits_content)    
    
    logits_content = k3 * logits_content + b3
    logits_content = logits_content + torch.tanh(c3) * torch.tanh(logits_content)    
      
    
  
   
    
    
    logits_content = logits_content.reshape((-1,num_points,pro_range))
  
    
    
    
    for i in range(len(self._filters) + 1):
      matrix = torch.nn.functional.softplus(self._matrices[i])
      logits = torch.matmul(matrix, logits)

      logits += self._biases[i]

      factor = torch.tanh(self._factors[i])
      
      logits += factor * torch.tanh(logits)
      
      
        
    return torch.abs(self._factors2[0]+1)*logits[channel_idx:channel_idx+1] + torch.abs(self._factors2[1]+0.01)*logits_content




  


  def _quantize(self, inputs, mode, device='cuda'):
    """Add noise or quantize."""
    if mode == "noise":
      noise = np.random.uniform(-0.5, 0.5, inputs.size())
      noise = torch.Tensor(noise).to(device)
      return inputs + noise

    if mode == "symbols":
      return RoundNoGradient.apply(inputs)

  def _likelihood(self, inputs, content):
    """Estimate the likelihood.
    
    Arguments:
      inputs: tensor with shape (points, channels).
    
    Return:
      likelihoods: tensor with shape(points, channels).
    """
    # reshape to (channels, 1, -1)
    inputs = inputs.permute(1, 0).contiguous()
    shape = inputs.size()# [channels, points]
    inputs = inputs.view(shape[0], 1, -1)

    """
    # other methods:
    # 1.
      inputs = inputs.view(-1, 1, channels)
      inputs = inputs.permute(2, 1, 0)
    # 2.
      inputs = inputs.permute(1, 0)
      shape = inputs.size()
      inputs = torch.reshape(inputs, (shape[0], 1, -1))
    """
    
    # Evaluate densities.
    lower = self._logits_cumulative(inputs - 0.5, content)
    upper = self._logits_cumulative(inputs + 0.5, content)

    sign = -torch.sign(torch.add(lower, upper))
    sign = sign.detach()# ? TODO
    likelihood = torch.abs(torch.sigmoid(sign * upper) - torch.sigmoid(sign * lower))
    # print('upper, lower')
    # print('upper', upper.mean().detach().cpu(), upper.max().detach().cpu(), upper.min().detach().cpu(), 
    # '\n', 'lower', lower.mean().detach().cpu(), lower.max().detach().cpu(), lower.min().detach().cpu(), )
    # reshape to (points, channels)
    likelihood = likelihood.view(shape)
    likelihood = likelihood.permute(1, 0)

    return likelihood

  def forward(self, inputs, content, training, device='cuda'):
    """Pass a tensor through the bottleneck.
    
    Arguments:
      inputs: The tensor to be passed through the bottleneck.
      
      Returns:
        values: `Tensor` with the shape as `inputs` containing the perturbed
        or quantized input values.
        likelihood: `Tensor` with the same shape as `inputs` containing the
        likelihood of `values` under the modeled probability distributions.
    """
    # print('input:::\n', inputs.mean().detach().cpu(), inputs.max().detach().cpu(), inputs.min().detach().cpu() )
    if training:
      outputs = self._quantize(inputs, "noise", device)
      #outputs = self._quantize(inputs, "symbols", device)
      
    else:
      outputs = self._quantize(inputs, "symbols", device)
      outputs = outputs.float()
      
    

    likelihood = self._likelihood(outputs, content)
    # print('likelihood')
    # print('likelihood:::\n', likelihood.mean().detach().cpu(), likelihood.min().detach().cpu())
    likelihood = Low_bound.apply(likelihood)
    #print(likelihood.sum(-1))

    return outputs, likelihood

  def _get_cdf(self, min_v, max_v, inputs, content, device='cpu'):
    """Get quantized cdf for compress/decompress.
    
    Arguments:
      inputs: integer numpy min_v, max_v.
    Return: 
      cdf with shape [1, channels, symbols] ? TODO check
    """
    import time
    
    
    #shape of cdf should be [C,1,N]
    a = np.reshape(np.arange(min_v, max_v + 1),[1, 1, max_v -min_v + 1])
    a = np.tile(a,[self._channels, 1, 1])
    # offset = 0.0
    a = torch.from_numpy(a).to(device)
    a = a.float().cuda()
    # print('a:',a.shape)

    # estimate pmf
    
    a_t = torch.cat((a-0.5, (a-0.5)[:,:,-1:]+1),-1)
    
    
    
    time_comp = time.time()
    l_u = self._logits_cumulative_compress(a_t, content)
    time_comp_end = time.time()
    
    self.time_comp_sum+=(time_comp_end-time_comp)
    #print('time_comp', time_comp_end-time_comp)
    
    
    
    # lower = self._logits_cumulative_compress(a - 0.5, content)
    # upper = self._logits_cumulative_compress(a + 0.5, content) 
    lower = l_u[:,:,:-1]
    upper = l_u[:,:,1:]
    
    

    


    sign = -torch.sign(torch.add(lower, upper))
    sign = sign.detach()# ? TODO
    likelihood = torch.abs(
      torch.sigmoid(sign * upper) - torch.sigmoid(sign * lower))
    likelihood = Low_bound.apply(likelihood)
    pmf = likelihood
    
    # from matplotlib import pyplot as plt 
    # plt.plot(np.arange(pmf.detach().cpu().numpy()[0,0].shape[0])+pmf.detach().cpu().numpy()[0,0][:].min(), pmf.detach().cpu().numpy()[0,0]) 
    # plt.show() 
    
    
    # temp=likelihood.detach().cpu()[0][0].numpy()
    # import matplotlib.pyplot as plt
    # plt.plot(np.arange(temp.shape[0])+min_v, temp) 
    
   
    
    time_cdf = time.time()
    

    # To Tensorflow
    #pmf_data = pmf.data.cpu().numpy()
    
    time_io = time.time()
    pmf_data = pmf.cpu().numpy()
    time_io_end = time.time()
    #print('time_io',time_io_end-time_io)
    
    
    
    
    time_cdf_end = time.time()
    self.time_cdf += (time_cdf_end - time_cdf)
    #print(self.time_cdf)   
    
    
    pmf_data = pmf_data/(np.sum(pmf_data,-1)[:,:,None])
    
    pmf_tf = tf.convert_to_tensor(pmf_data)
    
    
    # To quantized CDF.
    
    cdf = coder_ops.pmf_to_quantized_cdf(
      pmf_tf, precision = self._range_coder_precision)
    #print(pmf_tf.shape)
    time1_end=time.time()
    #print('pmf_to_quantized_cdf', time1_end-time1)
    
    # from matplotlib import pyplot as plt 
    # plt.plot(np.arange(cdf.numpy()[0,0].shape[0])+cdf.numpy()[0,0][:].min(), cdf.numpy()[0,0]) 
    # plt.show() 

    
    #cdf = tf.reshape(cdf, [1, self._channels, -1])  
    cdf = tf.transpose(cdf, perm=[1,0,2])
   
    return cdf


  def _get_cdf_decompress(self, min_v, max_v, content, channels_idx, device='cpu'):
    """Get quantized cdf for compress/decompress.
    
    Arguments:
      inputs: integer numpy min_v, max_v.
    Return: 
      cdf with shape [1, channels, symbols] ? TODO check
    """
    import time
    
    
    #shape of cdf should be [C,1,N]
    a = np.reshape(np.arange(min_v, max_v + 1),[1, 1, max_v -min_v + 1])
    a = np.tile(a,[1, 1, 1])
    # offset = 0.0
    a = torch.from_numpy(a).to(device)
    a = a.float().cuda()
    # print('a:',a.shape)

    # estimate pmf
    
    a_t = torch.cat((a-0.5, (a-0.5)[:,:,-1:]+1),-1)
    
    
    
    time_comp = time.time()
    l_u = self._logits_cumulative_decompress(a_t, content, channels_idx)
    time_comp_end = time.time()
    
    self.time_comp_sum+=(time_comp_end-time_comp)
    #print('time_comp', time_comp_end-time_comp)
    
    
    
    # lower = self._logits_cumulative_compress(a - 0.5, content)
    # upper = self._logits_cumulative_compress(a + 0.5, content) 
    lower = l_u[:,:,:-1]
    upper = l_u[:,:,1:]
    
    

    


    sign = -torch.sign(torch.add(lower, upper))
    sign = sign.detach()# ? TODO
    likelihood = torch.abs(
      torch.sigmoid(sign * upper) - torch.sigmoid(sign * lower))
    likelihood = Low_bound.apply(likelihood)
    pmf = likelihood
    
    # from matplotlib import pyplot as plt 
    # plt.plot(np.arange(pmf.detach().cpu().numpy()[0,0].shape[0])+pmf.detach().cpu().numpy()[0,0][:].min(), pmf.detach().cpu().numpy()[0,0]) 
    # plt.show() 
    
    
    # temp=likelihood.detach().cpu()[0][0].numpy()
    # import matplotlib.pyplot as plt
    # plt.plot(np.arange(temp.shape[0])+min_v, temp) 
    
   
    
    time_cdf = time.time()
    

    # To Tensorflow
    #pmf_data = pmf.data.cpu().numpy()
    pmf_data = pmf.cpu().numpy()
    
    time_cdf_end = time.time()
    self.time_cdf += (time_cdf_end - time_cdf)
    #print(self.time_cdf)   
    
    
    pmf_data = pmf_data/(np.sum(pmf_data,-1)[:,:,None])
    
    pmf_tf = tf.convert_to_tensor(pmf_data)
    
    
    # To quantized CDF.
    
    cdf = coder_ops.pmf_to_quantized_cdf(
      pmf_tf, precision = self._range_coder_precision)
    #print(pmf_tf.shape)
    time1_end=time.time()
    #print('pmf_to_quantized_cdf', time1_end-time1)
    
    # from matplotlib import pyplot as plt 
    # plt.plot(np.arange(cdf.numpy()[0,0].shape[0])+cdf.numpy()[0,0][:].min(), cdf.numpy()[0,0]) 
    # plt.show() 

    
    #cdf = tf.reshape(cdf, [1, self._channels, -1])  
    cdf = tf.transpose(cdf, perm=[1,0,2])
   
    return cdf











  def compress(self, inputs, content, device='cpu'):
    """Compress inputs and store their binary representations into strings.

    Arguments:
      inputs: `Tensor` with values to be compressed. Must have shape 
      [points, channels] (torch)
    Returns:
      compressed: String vector containing the compressed
        representation of each batch element of `inputs`. (numpy)
      min_v & max_v (numpy).
    """   
    # get symbols
    values = self._quantize(inputs, "symbols")
    values = values.detach()
    values = values.float()
    
    
    self._channels = values.shape[-1]

    # get range[min_v, max_v]
    min_v = torch.min(values.data)
    min_v = torch.floor(min_v)
    #min_v = min_v.short()
    #print('min:',min_v)
    max_v = torch.max(values.data)
    max_v = torch.ceil(max_v)
    #max_v = max_v.short()
    #print('max:', max_v)
    min_v_data = min_v.cpu().numpy().astype(np.int64)
    max_v_data = max_v.cpu().numpy().astype(np.int64)

    # early stop
    if min_v_data == max_v_data:
      strings = bytes(1)
      return strings, min_v_data, max_v_data

    # get cdf
    
    import time
    time1=time.time()
    cdf = self._get_cdf(min_v_data, max_v_data, inputs, content)
    time1_end=time.time()
    
    
    #print('cdf',self.time_cdf)    

    
    #print(cdf)
    
    
    # temp=cdf[0][0].numpy()
    # import matplotlib.pyplot as plt
    # plt.plot(np.arange(temp.shape[0]), temp)
    
    
    strings_list = []
    
    for i in range(self._channels):
    
    
        # To tensorflow
        values_data = values[:,i:i+1].detach().cpu().numpy()
        values_tf = tf.convert_to_tensor(values_data)# shape=[points, channels]
        values_tf = tf.reshape(values_tf, [-1, 1])# TODO: delete.
        values_tf = tf.cast(values_tf, tf.int16)
        values_tf -= min_v_data
        
        # range encode.
        strings = coder_ops.range_encode(
          values_tf, cdf[:,i:i+1], precision=self._range_coder_precision)
        
        strings_list.append(strings.numpy())
        

    # TODO: check numpy, torch, tf.
    return strings_list, min_v_data, max_v_data    
    
    

    
    
  def decompress(self, strings, min_v, max_v, shape, content, channels_idx, device='cpu'):
    """Decompress values from their compressed string representations.

    Arguments:
      strings: A string `Tensor` vector containing the compressed data.(numpy)
      min_v & max_v.(numpy)
      shape: A `Tensor` vector of int32 type. Contains the shape of the tensor to be
        decompressed, excluding the batch dimension. [points, channels] (numpy)

    Returns:
      The decompressed `Tensor`. (torch)  # TODO
    """   
    # To Tensorflow
    if min_v == max_v:
      values = torch.zeros(tuple(shape)).to(device)
      return values
  
  

    strings = tf.convert_to_tensor(strings, dtype=tf.string)
    #shape = tf.convert_to_tensor(shape)# [points, channels]
    #shape[1]=1
    shape = tf.convert_to_tensor(np.array([shape[0],1]).astype(np.int32))# [points, channels]
    
    
    cdf = self._get_cdf_decompress(min_v, max_v, content, channels_idx, device)


    # range decode
    values = coder_ops.range_decode(
        strings, shape, cdf, precision=self._range_coder_precision) + min_v
    values = tf.reshape(values, shape)
    values = tf.cast(values, tf.float32)
    values = values.numpy()
    # values = torch.from_numpy(values).to(device)
    # values_list.append(values)
    
    
        
    return values    
    
    
    
    
    
    
    
    
    
    
    


















