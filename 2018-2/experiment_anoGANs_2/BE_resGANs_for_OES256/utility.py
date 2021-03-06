import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec 
import numpy as np
from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix, precision_recall_curve, average_precision_score
import itertools
from sklearn_evaluation import plot
import scipy.io as sio

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()



def idx_shuffle(x) : 
    l = x.shape[0]
    idx = np.arange(l)
    np.random.shuffle(idx)
    shuffled_x = np.empty(x.shape)

    for i in range(l):
        shuffled_x[idx[i]] = x[i]
    
    return shuffled_x

def mnist_4by4_save(samples,path):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)    
    gs.update(wspace=0.05, hspace=0.05) #이미지 사이간격 조절
  
    for i, sample in enumerate(samples) :
        ax = plt.subplot(gs[i])
        plt.axis('off')    
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
   
        plt.imshow(sample.reshape(256, 256), cmap='Greys_r')
        plt.colorbar()
    plt.savefig(path+'.png', bbox_inches='tight')
    plt.close(fig)
   
    return None


def mnist_matlab_4by4_save(samples,path):
    size = 256
    mat = {'im_01':np.resize(samples[0],(size,size)),'im_02':np.resize(samples[1],(size,size)),'im_03':np.resize(samples[2],(size,size)),
          'im_04':np.resize(samples[3],(size,size)),'im_05':np.resize(samples[4],(size,size)),'im_06':np.resize(samples[5],(size,size)),
          'im_07':np.resize(samples[6],(size,size)),'im_08':np.resize(samples[7],(size,size)),'im_09':np.resize(samples[8],(size,size)),
          'im_10':np.resize(samples[9],(size,size)),'im_11':np.resize(samples[10],(size,size)),'im_12':np.resize(samples[11],(size,size)),
          'im_13':np.resize(samples[12],(size,size)),'im_14':np.resize(samples[13],(size,size)),'im_15':np.resize(samples[14],(size,size)),
          'im_16':np.resize(samples[15],(size,size))}
    sio.savemat(path + '.mat', mat)
    
    
    return None
    
def gan_loss_graph_save(G_loss,D_loss,path):
    x1 = range(len(G_loss))
    x2 = range(len(D_loss))
      
    y1 = G_loss
    y2 = D_loss
  
    fig = plt.figure()  
    plt.plot(x1,y1,label='G_loss') 
    plt.plot(x2,y2,label='D_loss') 
  
    plt.xlabel('weight per update')
    plt.ylabel('loss')             
    plt.legend()              
    plt.grid(True)
    plt.tight_layout()
  
    plt.savefig(path)     
    plt.close(fig)

    return None


def my_roc_curve(normal, anomaly, path, name, is_True_one = True) : 


    pred = np.concatenate((np.reshape(normal,-1), np.reshape(anomaly,-1)), axis=0)

        
    label_normal = np.zeros(np.array(normal).shape[0])
    label_anomalous = np.ones(np.array(anomaly).shape[0])
    test = np.concatenate((label_normal, label_anomalous), axis=0)

    
    fpr = dict()
    tpr = dict()
    fpr, tpr, thresh = roc_curve(test, pred)
    roc_auc = auc(fpr, tpr)

    fig = plt.figure()
    plt.plot(fpr, tpr, color='darkorange',
             lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.01])
    plt.xlabel('False Positive Rate(1-specificity)')
    plt.ylabel('True Positive Rate(sensitivity)')
    plt.title('ROC Curve - '+ name)
    plt.legend(loc="lower right")

    plt.savefig(path)     
    plt.close(fig)
    
    pred_class = pred >= thresh[np.argmax((1-fpr)**2 +tpr**2)]
 
    fig1 = plt.figure()
    plot_confusion_matrix(confusion_matrix(test, pred_class), classes='', normalize=True,
                          title='Normalized confusion matrix - ' + name)
    plt.savefig(path+'_confusion matrix_normal')     
    plt.close(fig1)
    
    fig2 = plt.figure()
    plot_confusion_matrix(confusion_matrix(test, pred_class), classes='',
                          title='confusion matrix - ' + name)
    plt.savefig(path+'_confusion matrix')     
    plt.close(fig2)
    


def my_hist(test_normal, test_anomaly,train_normal, path, name) : 

    fig = plt.figure()
    plt.hist(np.reshape(test_normal,-1), bins='auto', alpha=0.6, label="test normal",normed=1)  
    plt.hist(np.reshape(test_anomaly,-1), bins='auto', alpha=0.6, label="test anomaly",normed=1)  
    plt.hist(np.reshape(train_normal,-1), bins='auto', alpha=0.6, label="train normal",normed=1)  

    plt.xlabel('score value') 
    plt.title('PDF - '+ name)
    plt.legend()

    plt.savefig(path)     
    plt.close(fig)





