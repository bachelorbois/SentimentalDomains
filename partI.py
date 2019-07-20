from talos.metrics.keras_metrics import fmeasure_acc
import json
from src.neural import SentenceSentimentTopicGenerator, SentimentSentenceRNN
from keras.callbacks import TensorBoard, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import confusion_matrix
import numpy as np
from argparse import ArgumentParser
import glob 

params = {
    'topic_embed': 50, 
    'rnn_layer_sizes': [64],  
    'layer_type': 'LSTM', 
    'batch_size': 256, 
    'optimizer': 'nadam',
    'vocab_size': 27112,
    'sent_embed': 50,
    'num_topics': 12,
    'dense_sizes': [128, 64],
    'embedding': './data/embed_mat.npy',
    'ablate': False
}

vocab_sizes = {
    'full_data': 27112,
    'JJ': 24225,
    'SW': '26962'
}

def run(params, args):
    print("Initializing data....")
    vocab = "../data_phase3_partI/word2int.json"
    if '_JJ' in args.train:
        vocab = "../data_phase3_partI/word2int_JJ.json"
        params['vocab_size'] = vocab_sizes['JJ']
        params['embedding'] = './data/embedding_JJ.npy'
    elif '_SW' in args.train:
        vocab = "../data_phase3_partI/word2int_STOP_WORDS.json"
        params['vocab_size'] = vocab_sizes['SW']
        params['embedding'] = './data/embedding_SW.npy'
    
    params['ablate'] = args.D
    # Load the main data files into a generator
    train_gen = SentenceSentimentTopicGenerator(args.train, vocab=vocab, review_len=500, seq_len=None, batch_size=params["batch_size"])
    # Load the main test file into a generator
    val_gen = SentenceSentimentTopicGenerator(args.val, vocab=vocab, review_len=500, seq_len=None, batch_size=128)

    # Init the model using {'topic_embed': 7, 'layer_s1': 64, 'layer_s2': 128, 'layer_type': 'GRU', 'batch_size': 32, 'optimizer': 'adam'}
    model = SentimentSentenceRNN(
        num_classes=1,
        params=params
    )
    # Compile model
    model.compile(params["optimizer"], "binary_crossentropy", metrics=["acc", fmeasure_acc]) 

    # history = {'loss': [], 'acc': [], 'fmeasure_acc': [], 'val_loss': [], 'val_acc': [], 'val_fmeasure_acc': []}

    topic_t = np.array([[1],[2],[3],[4],[5],[6],[7],[8],[9],[10],[11],[12]], dtype='int32') 
    folder = glob.glob("./logs/*")
    if not args.D:
        tb = TensorBoard(log_dir=f'./logs/SentLogs_{len(folder)-1}', embeddings_freq=1, embeddings_layer_names=["TopicEmbedding"], embeddings_metadata='../data_phase3_partI/metadata.tsv', embeddings_data=[topic_t, topic_t], update_freq=1000)
    else:
        tb = TensorBoard(log_dir=f'./logs/SentLogs_{len(folder)-1}', update_freq=1000)
    lr = ReduceLROnPlateau(monitor='val_acc', factor=0.2,
                            patience=5, min_lr=0.0001)
    checkpoint = ModelCheckpoint('./models/sentRNN{epoch:20d}.h5', monitor='val_acc')

    # Start a training loop {5 Epochs train 1 Epoch test} X 10
    
    _ = model.fit_generator(train_gen, val_gen,
                        epochs=10, use_multiprocessing=False,
                        workers=1, callbacks=[tb, lr, checkpoint], verbose=1)
    

def eval_model(params, args):
    print('Initializing variables....')
    f = open('Evaluation.csv', 'a+')
    if '_JJ' in args.train:
        vocab = "../data_phase3_partI/word2int_JJ.json"
        params['vocab_size'] = vocab_sizes['JJ']
        params['embedding'] = './data/embedding_JJ.npy'
    elif '_SW' in args.train:
        vocab = "../data_phase3_partI/word2int_STOP_WORDS.json"
        params['vocab_size'] = vocab_sizes['SW']
        params['embedding'] = './data/embedding_SW.npy'
    
    params['ablate'] = args.D
    # for end in ['', '_JJ', '_SW']:
    test_gen = SentenceSentimentTopicGenerator(args.test, vocab=vocab, review_len=500, seq_len=None, batch_size=128, ablate=args.D)
    model = SentimentSentenceRNN(
        num_classes=1,
        params=params
    )
    model.load('./models/sentRNN{epoch:20d}.h5'.format(epoch=10))
    metrics = model.evaluate_generator(test_gen, use_multiprocessing=False, workers=1, verbose=1) 
    f.write('sentRNN'+str(metrics)+' \n')
    f.close()

def getclasses(gen):
    classes = []
    for i in range(len(gen)):
        _, y = gen[i]
        classes.extend(y)
    return classes

def conf_mat(params):
    print("Get ready for some action...")
    f = open('ConfMat.csv', 'a+')
    # f.write('model,tn,fp,fn,tp \n')
    # for end in ['', '_JJ', '_SW']:
    test_gen = SentenceSentimentTopicGenerator("../data_phase3_partI/test.txt", vocab="../data_phase3_partI/word2int.json", review_len=500, seq_len=None, batch_size=128)
    model = SentimentSentenceRNN(
        num_classes=1,
        params=params
    )
    model.load('./models/sentRNN_Topic{epoch:20d}.h5'.format(epoch=10))
    preds = model.predict_generator(test_gen, use_multiprocessing=False, workers=1, verbose=1)
    test_gen.on_epoch_end()
    print(preds.shape)
    tn, fp, fn, tp = confusion_matrix(getclasses(test_gen),np.around(preds)).ravel()
    f.write('sentRNN_Topic,'+str(tn)+','+str(fp)+','+str(fn)+','+str(tp)+' \n')
    f.close()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-train', type=str, required=True, help='The dataset used to do training')
    parser.add_argument('-val', type=str, required=True, help='The dataset used to do validation')
    parser.add_argument('-test', type=str, required=True, help='The dataset used to do testing')
    parser.add_argument('-D', action='store_true', help='Set this flag to do domain ablation')
    args = parser.parse_args()
    run(params, args)
    eval_model(params, args)
