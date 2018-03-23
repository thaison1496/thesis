import codecs
import numpy as np

def predict_to_file(predicts, tests, alphabet_tag, output_file):
    ###
    print(alphabet_tag.instances)
    ###
    print(predicts.shape)
    with codecs.open(output_file, 'w', 'utf-8') as f:
        for i in range(len(predicts)):
            for j in range(len(predicts[i])):
                predict = alphabet_tag.get_instance(predicts[i][j])
                if predict == None:
                    predict = alphabet_tag.get_instance(predicts[i][j] + 1)

                test = alphabet_tag.get_instance(np.argmax(tests[i][j]))
                if test == None:
                    # print(j)
                    break
                # print(predict)
                # print(tests[i][j])
                # print(np.argmax(tests[i][j]))
                # print(test)
                f.write('_' + ' ' + predict + ' ' + test + '\n')
            f.write('\n')