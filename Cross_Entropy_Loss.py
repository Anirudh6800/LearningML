import numpy as np

# solftmax_output array
softmax_output = np.array([[0.7, 0.1, 0.2],
                          [0.1, 0.5, 0.4],
                          [0.02, 0.09, 0.08]])
# true_labels array
class_target = [0,1,1]


print(softmax_output[[0, 1, 2], class_target])

print(-np.log(softmax_output[range(len(softmax_output)), class_target]))
neg_log_softmax = -np.log(softmax_output[range(len(softmax_output)), class_target])
print(neg_log_softmax)
average_loss = np.mean(neg_log_softmax)
print(average_loss)




# Cross-entropy loss function
def cross_entropy_loss(softmax_output, class_target):
    """
    Compute the cross-entropy loss between the softmax output and the true labels.
    Args:
        softmax_output: Softmax output array (predicted probabilities)
        class_target: True labels (one-hot encoded)
    Returns:
        Cross-entropy loss value
        


        """
