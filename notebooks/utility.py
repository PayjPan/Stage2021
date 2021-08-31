import matplotlib.pyplot as plt

from sklearn.metrics import plot_confusion_matrix, classification_report, confusion_matrix


def mesure_clas(model, X_test, y_test):
    """ Return some mesures on the classifier 'model'
    """

    y_test_pred = model.predict(X_test)
    results = classification_report(y_test, 
                                    y_test_pred,
                                    output_dict=True)

    disp = plot_confusion_matrix(model,
                                 X_test,
                                 y_test,
                                 cmap=plt.cm.Blues)
    _ = disp.ax_.set_title('Confusion matrix')
    plt.show()


    print("{:^12} {:^12} {:^12} {:^12}".format('Sensitivity', 
                                               'Specificity', 
                                               'Precision', 
                                               'Accuracy'))
    print("{:^12.2f} {:^12.2f} {:^12.2f} {:^12.2f}".format(results['0']['recall'], 
                                                           results['1']['recall'], 
                                                           results['1']['precision'],
                                                           results['accuracy']))

def mesure_clas_list(model_list, X_test, y_test, name_list=['model']):
    """ Return some mesures on the classifiers in the list 'model_list'
        Plot only the confusion matrix of the first one
    """
    
    results_list = []
    for model in model_list:
        y_test_pred = model.predict(X_test)
        results_list.append(classification_report(y_test, 
                                                  y_test_pred,
                                                  output_dict=True))

    disp = plot_confusion_matrix(model_list[0],
                                 X_test,
                                 y_test,
                                 cmap=plt.cm.Blues)
    _ = disp.ax_.set_title('Confusion matrix')
    plt.show()


    size = max([len(name) for name in name_list])
    print("{:^{}} {:^12} {:^12} {:^12} {:^12}".format('',
                                                      size,
                                                      'Sensitivity', 
                                                      'Specificity', 
                                                      'Precision', 
                                                      'Accuracy'))
    for i, results in enumerate(results_list):
        print("{:>{}} {:^12.2f} {:^12.2f} {:^12.2f} {:^12.2f}".format(name_list[i],
                                                                 size,
                                                                 results['0']['recall'], 
                                                                 results['1']['recall'], 
                                                                 results['1']['precision'],
                                                                 results['accuracy']))
