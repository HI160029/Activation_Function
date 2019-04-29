
import numpy as np
import tensorflow as tf
import pickle
from tensorflow.contrib.layers import flatten
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB as nb
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import LabelEncoder, LabelBinarizer

# load file
pickle_file = open( "mnist_a2j.pickle", "rb" )
data = pickle.load(pickle_file)
test_labels = data["test_labels"]
train_labels = data["all_labels"]
test_dataset = data["test_dataset"]
train_dataset = data["all_dataset"]
del data
pickle_file.close()

print("Training size: ", train_dataset.shape)
print("Training labels: ", train_labels.shape)
print("Test size: ", test_dataset.shape)
print("Test labels: ", test_labels.shape)

# Activation function
def ReLU (x): # ReLU
  return tf.nn.relu(x)

def Swish (x):
  return x*tf.sigmoid(x)

def Tanh (x):
  return tf.nn.tanh (x)

def LReLU_01 (x):
  return tf.maximum(x,x*0.01)

def LReLU_25 (x):
  return tf.maximum(x,x*0.25)

def PReLU(x): # PReLU
  a5=tf.Variable(initial_value=0.25,trainable=True,name='alpha5')
  return tf.maximum(0.0, x) + a5 * tf.minimum(0.0, x)

def Softplus (x):
  return tf.nn.softplus(x)

def ELU (x):
  return tf.nn.elu (x)

def FReLU (x): # Non-trainable
  return tf.nn.relu(x)-0.398

def FReLU_TRAIN (x): # Trainable FRELU # Initializa=-0.398
  bs=tf.Variable(initial_value=-0.398,trainable=True,name='bias')
  return tf.nn.relu(x)+bs 

def FTS (x): # FTS [T=-0.20 (Fix)] 
  return (tf.nn.relu(x)*tf.sigmoid(x))-0.20 # trainable FTS

def FTS_train (x): # FTS [T=-0.20 (Train)] 
  T_train=tf.Variable(initial_value=-0.20,trainable=True,name='threshold')
  activation=(tf.nn.relu(x)*tf.sigmoid(x))+T_train
  return activation

def FTS1 (x): # FTS [T=-0.20 (Train), a=0.342061 (Train)] 
  T1=tf.Variable(initial_value=-0.20,trainable=True,name='threshold1')
  a1=tf.Variable(initial_value=0.34206097,trainable=True,name='alpha1')
  activation=(tf.nn.relu(x+a1)*tf.sigmoid(x+a1))+T1 
  return activation

def FTS2 (x): # FTS [T=-0.20 (Train), a=0.0 (Train)] 
  T2=tf.Variable(initial_value=-0.20,trainable=True,name='threshold2')
  a2=tf.Variable(initial_value=0.00,trainable=True,name='alpha2')
  activation=(tf.nn.relu(x+a2)*tf.sigmoid(x+a2))+T2
  return activation

def FTS3 (x): # FTS [T=-0.20 (Fix), a=0.342061 (Train)] 
  T3=-0.20
  a3=tf.Variable(initial_value=0.34206097,trainable=True,name='alpha3')
  activation=(tf.nn.relu(x+a3)*tf.sigmoid(x+a3))+T3
  return activation

def FTS4 (x): # FTS [T=-0.20 (Fix), a=0.0 (Train)] 
  T4=-0.20
  a4=tf.Variable(initial_value=0.0,trainable=True,name='alpha4')
  activation=(tf.nn.relu(x+a4)*tf.sigmoid(x+a4))+T4
  return activation

def FTS5 (x): # FTS [T=0.0 (Train), a=0.0 (Train)] 
  T5=tf.Variable(initial_value=0.00,trainable=True,name='threshold3')
  a5=tf.Variable(initial_value=0.00,trainable=True,name='alpha5')
  activation=(tf.nn.relu(x+a5)*tf.sigmoid(x+a5))+T5
  return activation

functions={
    'ReLU':ReLU,
    'Swish':Swish,
    'Tanh':Tanh,
    'LReLU_01':LReLU_01,
    'LReLU_25':LReLU_25,
    'PReLU':PReLU,
    'Softplus':Softplus,
    'ELU':ELU,
    'FReLU':FReLU,
    'FReLU_TRAIN':FReLU_TRAIN,
    'FTS':FTS,
    'FTS_train':FTS_train,
    'FTS1':FTS1,
    'FTS2':FTS2,
    'FTS3':FTS3,
    'FTS4':FTS4,
    'FTS5':FTS5,
    'PReLU':PReLU
}


num_labels = train_labels.shape[1]
num_data = train_labels.shape[0]
EPOCHS = 100
BATCH_SIZE = 64
learning_rate = 0.0001
dropout_rate = 0.95

# Network Parameters
n_input = 784  
n_hidden_1 = 784  # 1st layer number of features
n_hidden_2 = 512  # 1st layer number of features
n_hidden_3 = 256  # 2nd layer number of features
n_hidden_4 = 128  # 3nd layer number of features
n_hidden_5 = 64  # 4nd layer number of features
n_hidden_6 = 32  # 4nd layer number of features
n_classes = 10  # 43 total classes 

def get_inputs():
  x = tf.placeholder(tf.float32, shape=[None, n_input])
  y = tf.placeholder(tf.float32, shape=[None, n_classes])
  keep_prob = tf.placeholder(tf.float32) # probability to keep units
  return x,y,keep_prob


def NN(x,y,keep_prob,func):
    fc=functions[func]    

    x = flatten(x)
    
    # Layer 1
    w1 = tf.get_variable(shape=[n_input, n_hidden_1], initializer=tf.contrib.layers.xavier_initializer(), name='weights1')
    b1 = tf.Variable(tf.constant(0.0, shape=[n_hidden_1], dtype=tf.float32), trainable=True, name='biases1'),   
    x = tf.add(tf.matmul(x, w1), b1)
    x = fc(x)
    x = tf.nn.dropout(x, keep_prob)

    # Layer 2
    w2 = tf.get_variable(shape=[n_hidden_1, n_hidden_2], initializer=tf.contrib.layers.xavier_initializer(), name='weights2')
    b2 = tf.Variable(tf.constant(0.0, shape=[n_hidden_2], dtype=tf.float32), trainable=True, name='biases2'), 
    x = tf.add(tf.matmul(x, w2), b2)
    x = fc(x)
    x = tf.nn.dropout(x, keep_prob)

     # Layer 3
    w3 = tf.get_variable(shape=[n_hidden_2, n_hidden_3], initializer=tf.contrib.layers.xavier_initializer(), name='weights3')
    b3 = tf.Variable(tf.constant(0.0, shape=[n_hidden_3], dtype=tf.float32), trainable=True, name='biases3'), 
    x = tf.add(tf.matmul(x, w3), b3)
    x = fc(x)
    x = tf.nn.dropout(x, keep_prob)

     # Layer 4
    w4 = tf.get_variable(shape=[n_hidden_3, n_hidden_4], initializer=tf.contrib.layers.xavier_initializer(), name='weights4')
    b4 = tf.Variable(tf.constant(0.0, shape=[n_hidden_4], dtype=tf.float32), trainable=True, name='biases4'), 
    x = tf.add(tf.matmul(x, w4), b4)
    x = fc(x)
    x = tf.nn.dropout(x, keep_prob)

     # Layer 5
    w5 = tf.get_variable(shape=[n_hidden_4, n_hidden_5], initializer=tf.contrib.layers.xavier_initializer(), name='weights5')
    b5 = tf.Variable(tf.constant(0.0, shape=[n_hidden_5], dtype=tf.float32), trainable=True, name='biases5'), 
    x = tf.add(tf.matmul(x, w5), b5)
    x = fc(x)
    x = tf.nn.dropout(x, keep_prob)

     # Layer 6
    w6 = tf.get_variable(shape=[n_hidden_5, n_hidden_6], initializer=tf.contrib.layers.xavier_initializer(), name='weights6')
    b6 = tf.Variable(tf.constant(0.0, shape=[n_hidden_6], dtype=tf.float32), trainable=True, name='biases6'), 
    x = tf.add(tf.matmul(x, w6), b6)
    x = fc(x)
    x = tf.nn.dropout(x, keep_prob)

    # Layer 7
    w7 = tf.get_variable(shape=[n_hidden_6, n_classes], initializer=tf.contrib.layers.xavier_initializer(), name='weights7')
    b7 = tf.Variable(tf.constant(0.0, shape=[n_classes], dtype=tf.float32), trainable=True, name='biases7'), 
    logits = tf.add(tf.matmul(x, w7), b7)

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)
    loss_operation = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate)
    training_operation = optimizer.minimize(loss_operation)
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return training_operation, loss_operation, accuracy_operation, logits


def evaluate(X_data, y_data):
    num_examples =18724
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy,loss = sess.run([accuracy_operation, loss_operation], feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples,loss
  
#'ReLU','Swish','Tanh','LReLU_01','LReLU_25','PReLU','Softplus','ELU','FReLU','FReLU_TRAIN','FTS','FTS_train','FTS1','FTS2','FTS3','FTS4','FTS5'
for func in ['LReLU_01','LReLU_25','PReLU','Softplus','ELU','FReLU','FReLU_TRAIN','FTS_train','FTS1','FTS2','FTS3','FTS4']:
  tf.reset_default_graph()
  x,y,keep_prob=get_inputs()
  training_operation, loss_operation, accuracy_operation, logits = NN(x,y,keep_prob,func)
  print (func)
  
  macro_auc = dict()
  micro_auc = dict()
  weighted_auc = dict()
  samples_auc = dict()

  for Run in range (5):
      with tf.Session() as sess:
          sess.run(tf.global_variables_initializer())
          num_examples = 529114
          
          Total_batch=int(num_examples/BATCH_SIZE)
          #print("Total batch = ", Total_batch)
          
          print()
          for i in range(EPOCHS):
              for offset in range(0, num_examples, BATCH_SIZE):
                  end = offset + BATCH_SIZE
                  batch_x, batch_y = train_dataset[offset:end], train_labels[offset:end]
                  sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: dropout_rate})
              
              if i==99:
                  print("EPOCH {}".format(i+1), "| Activation Function: ",func)
                  train_loss = sess.run(loss_operation, feed_dict={x: batch_x, y: batch_y, keep_prob:1.0})
                  train_accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob:1.0})
                  test_accuracy, test_loss = evaluate(test_dataset, test_labels)

                  print("Train Loss = {:.4f}".format(train_loss), 
                    "| Train Accuracy = {:.4f}".format(train_accuracy),
                    "| Test Loss = {:.4f}".format(test_loss),
                    "| Test Accuracy = {:.4f}".format(test_accuracy))
                  print()

          # Confusion matrix, ROC, AUC
          y_test=sess.run(logits,feed_dict={x:test_dataset,keep_prob:1.0})
          true_class = np.argmax(test_labels,1)
          predicted_class=np.argmax(y_test,1)
          #cm=confusion_matrix(true_class, predicted_class)
          #print(cm)
          #print("\n")
          #print(classification_report(true_class, predicted_class))
          
          #Source: https://scikit-learn.org/stable/modules/model_evaluation.html
          
          # ROC & AUc
          lb = LabelBinarizer()
          lb.fit(predicted_class)
          y_test_tran = lb.transform(true_class)
          y_pred_tran = lb.transform(predicted_class)
          macro_auc[Run] = roc_auc_score(y_test_tran, y_pred_tran, average="macro")
          micro_auc[Run] = roc_auc_score(y_test_tran, y_pred_tran, average="micro")
          weighted_auc[Run] = roc_auc_score(y_test_tran, y_pred_tran, average="weighted")
          samples_auc[Run] = roc_auc_score(y_test_tran, y_pred_tran, average="samples")
          print("Macro AUC = %.4f" %(macro_auc[Run]))
          print("Micro AUC = %.4f" %(micro_auc[Run]))
          print("Weighted AUC = %.4f" %(weighted_auc[Run]))
          print("Sample AUC = %.4f"%(samples_auc[Run]))

          print("----------------------------------------------------------------------------------------------------------------------------------")

          print("\n")
          
  print("----------------------------------------------------------------------------------------------------------------------------------")  
  mean_macro_auc = sum(macro_auc.values())/5
  mean_micro_auc = sum(micro_auc.values())/5
  weighted_macro_auc = sum(weighted_auc.values())/5
  samples_macro_auc = sum(samples_auc.values())/5
  print("Mean Macro AUC = %.4f" %mean_macro_auc)
  print("Mean Micro AUC = %.4f" %mean_micro_auc)
  print("Weighted Macro AUC = %.4f" %weighted_macro_auc)
  print("Samples Macro AUC = %.4f" %samples_macro_auc)

print("---------------End---------------------------------------------------------------------------------------------------------------------------------")
print("\n")