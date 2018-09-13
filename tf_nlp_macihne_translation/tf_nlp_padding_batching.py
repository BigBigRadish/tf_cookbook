# -*- coding: utf-8 -*-
'''
Created on 2018年9月13日

@author: Zhukun Luo
Jiangxi university of finance and economics
'''
import tensorflow as tf
#将所有数据读入内存，使用dataset从磁盘动态读取数据
MAX_LEN=50#限定句子的最大单词数
SOS_ID=1#目标语言<sos>的 id
#使用Dataset从一个文件中读取一个语言的数据
#数据的格式为每行一句话，单词已经转化为单词编号
def MakeDataset(file_path):
    dataset=tf.data.TextLineDataset(file_path)
    #根据空格将单词编号切分开并放入一个一维向量
    dataset=dataset.map(lambda string:tf.string_split([string]).values)
    #将字符串形式的单词编号转化为整数
    dataset=dataset.map(lambda string:tf.string_to_number(string, tf.int32))
    #统计每个句子的单词数量，并于句子内容一起放入Dataset
    dataset=dataset.map(lambda x: (x,tf.size(x)))
    return dataset
#从源文件src_path和目标语言文件trg_path中分别读取文件
#，并进行padding和bating操作
def MakeSrcTrgDataset(src_path,trg_path,batch_size):
    #分别读取语言数据和目标语言数据
    scr_data=MakeDataset(src_path)
    trg_data=MakeDataset(trg_path)
    #通过zip操作将两个dataset合并为一个dataset。现在每个Dataset中每一项数据ds是由四个张量组成
    #ds[0][0]是远举子，ds[0][1]是源句子长度，ds[1][0]是目标句子，ds[1][1]是目标句子长度
    dataset=tf.data.Dataset.zip((scr_data,trg_data))
    def FilterLenth(src_tuple,trg_tuple):
        ((src_input,src_len),(trg_label,trg_len))=(src_tuple,trg_tuple)
        src_len_ok=tf.logical_and(tf.greater(src_len,1),tf.less_equal(src_len,MAX_LEN))
        trg_len_ok=tf.logical_and(tf.greater(trg_len,1),tf.less_equal(trg_len,MAX_LEN))
        return tf.logical_and(src_len_ok,trg_len_ok)
    dataset=dataset.filter(FilterLenth)
    def MakeTrgInput(src_tuple,trg_tuple):
        ((src_input,src_len),(trg_label,trg_len))=(src_tuple,trg_tuple)
        trg_input=tf.concat([[SOS_ID],trg_label[:-1]],axis=0)
        return ((src_input,src_len),(trg_input,trg_label,trg_len))
    dataset=dataset.map(MakeTrgInput)
    #随即打乱训练数据
    dataset=dataset.shuffle(10000)
    #规定填充后输出的数据维度
    padded_shapes=((tf.TensorShape([None]),tf.TensorShape([])),(tf.TensorShape([None]),tf.TensorShape([None]),tf.TensorShape([])))
    #调用padded_batch方法进行batching操作
    batched_dataset=dataset.padded_batch(batch_size, padded_shapes)
    return batched_dataset
###############################################################################################################################################################
SRC_TRAIN_DATA='../dataset/ptb.train.en'
TRG_TRAIN_DATA='../dataset/ptb.train.zh'
CHECKPOINT_PATH='../data/seq2seq_ckpt'#checkpoint保存路劲
HIDDEN_SIZE=1024#lstm隐藏层规模
NUM_LAYERS=2#lstm的层数
SRC_VOCAB_SIZE=10000#源语言词汇表大小
TRG_VOCAB_SIZE=4000#目标语言词汇表大小
BATCH_SIZE=100#batch大小
NUM_EPPOCH=5#训练轮数
KEEP_PROB=0.8#节点不被dropout的概率
MAX_GRAD_NORM=5#用于控制梯度膨胀的梯度大小上限
SHARE_EMB_AND_SOFTMAX=True
#定义NMTMODEL类来描述模型
class NMTModel(object):
    #在模型的初始化函数中定义模型要用到的变量
    def __init__(self):
        #定义编码器与解码器使用的latm结构
        self.enc_cell=tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE) for _ in range(NUM_LAYERS)])
        self.dec_cell=tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE) for _ in range(NUM_LAYERS)])
        #为源语言和目标语言分别定义词向量
        self.src_embdding=tf.get_variable('src_emb',[SRC_VOCAB_SIZE,HIDDEN_SIZE])
        self.trg_embdding=tf.get_variable('trg_emb',[TRG_VOCAB_SIZE,HIDDEN_SIZE])
        #定义softmax层的变量
        if SHARE_EMB_AND_SOFTMAX:
            self.softmax_weight=tf.transpose(self.trg_embdding)
        else:
            self.softmax_weight=tf.get_variable('weight',[HIDDEN_SIZE,TRG_VOCAB_SIZE])
        self.softmax_bias=tf.get_variable('softmax_bias',[TRG_VOCAB_SIZE])
        #在forward函数中定义模型的前向计算图
        #src_input,src_size,trg_input,trg_label,trg_size分别是上面MakeSrcTrgDataset产生的5个张量
    def forward(self,src_input,src_size,trg_input,trg_label,trg_size):
        batch_size=tf.shape(src_input)[0]    
        #将输入和输出单词编号转为词向量
        src_emb=tf.nn.embedding_lookup(self.src_embdding,src_input)
        trg_emb=tf.nn.embedding_lookup(self.trg_embdding,trg_input)
        #在词向量上进行dropout
        src_emb=tf.nn.dropout(src_emb,KEEP_PROB)
        trg_emb=tf.nn.dropout(trg_emb,KEEP_PROB)
        #使用dynamic_rnn构造编码器，编码器读取源句子每个位置的词向量，输出最后一步的隐藏状态env_state.因为编码器是一个双层lstm，因此enc_state
        #是一个包含两个lstmstateTuple类的tuple，每个对应编码器中一层的状态
        #enc_outputs是顶层lstm在每一步的输出，它的维度是【batch_size,max_time,HIDDEN_SIZE】，SEQ2SEQ不需要用到enc_output,但是attention会用到他
        with tf.variable_scope('encoder'):
            enc_ouputs,enc_state=tf.nn.dynamic_rnn(self.enc_cell,src_emb,src_size,dtype=tf.float32)
        #使用dynamic_rnn构造解码器
        #解码器读取目标句子每个位置的词向量，输出的dec_outputs为每一步顶层lstm的输出。它的维度是【batch_size,max_time,HIDDEN_SIZE】，
        #initial_state=enc_state表示用编码器的输出来初始化第一部的隐藏状态
        with tf.variable_scope('decoder'):
            dec_ouputs,enc_state=tf.nn.dynamic_rnn(self.dec_cell,trg_emb,src_size,dtype=tf.float32)
        #计算解码器每一步的log perplexity
        output=tf.reshape(dec_ouputs,[-1,HIDDEN_SIZE])
        logits=tf.matmul(output, self.softmax_weight)+self.softmax_bias
        loss=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.reshape(trg_label,[-1]),logits=logits)
        #在计算平均损失时，需要将填充位置的权重设为0，以避免无效位置的预测干扰模型的训练
        label_weights=tf.sequence_mask(trg_size, maxlen=tf.shape(trg_label)[1], dtype=tf.float32)
        label_weights=tf.reshape(label_weights,[-1])
        cost=tf.reduce_sum(loss*label_weights)
        cost_per_token=cost/tf.reduce_sum(label_weights)
        #定义反向传播操作。
        trainable_variables=tf.trainable_variables()
        #控制梯度大小，定义优化方法和步骤
        grads=tf.gradients(cost/tf.to_float(batch_size),trainable_variables)
        grads,_=tf.clip_by_global_norm(grads,MAX_GRAD_NORM)
        optimizer=tf.train.GradientDescentOptimizer(learning_rate=1.0)
        train_op=optimizer.apply_gradients(zip(grads,trainable_variables))
        return cost_per_token,train_op
#使用给定的模型上训练一个epoch,并返回全局步数
#将训练两百步保存一个checkpoint
def run_epoch(session,cost_op,train_op,saver,step):
    #训练一个epoch，重复训练步骤直至遍历完dataset中所有数据
    while True:
        try:
            #运行train_op并计算损失值
            cost,_=session.run([cost_op,train_op])
            if step%10==0:
                print('after %d steps,per token cost is%.3f'%(step,cost))
            #每两百步保存一个checkpoint
            if step%200==0:
                saver.save(session,CHECKPOINT_PATH,global_step=step)
            step+=1
        except tf.errors.OutOfRangeError:
            break
    return step
def main(): 
    #定义初始化函数
    initializer=tf.random_uniform_initializer(-0.05,0.05)
    #定义循环神经网络模型
    with tf.variable_scope('nmt_model',reuse=None,initializer=initializer):
        train_model=NMTModel()
    #定义输入数据
    data=MakeSrcTrgDataset(SRC_TRAIN_DATA, TRG_TRAIN_DATA, BATCH_SIZE)
    iterator=data.make_initializable_iterator()
    (src,src_size),(trg_input,trg_label,trg_size)=iterator.get_next()
    #定义前向计算图
    cost_op,train_op=train_model.forward(src,src_size,trg_input,trg_label,trg_size)
    #训练模型
    saver=tf.train.Saver()
    step=0
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(NUM_EPPOCH):
            print('in iteration:%d'%(i+1))
            sess.run(iterator.initializer)
            step=run_epoch(sess, cost_op, train_op, saver, step)
if __name__ == '__main__':
    main()
   
        
